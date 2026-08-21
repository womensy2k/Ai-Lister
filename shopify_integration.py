import html
import mimetypes
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

SHOPIFY_API_VERSION = "2026-07"
load_dotenv()


def _shop():
    value = os.getenv("SHOPIFY_SHOP", "").strip()
    value = value.replace("https://", "").replace("http://", "").rstrip("/")
    if value and not value.endswith(".myshopify.com"):
        value += ".myshopify.com"
    return value


def _token():
    shop = _shop()
    cid = os.getenv("SHOPIFY_CLIENT_ID")
    secret = os.getenv("SHOPIFY_CLIENT_SECRET")

    if not shop:
        raise RuntimeError("SHOPIFY_SHOP is missing from .env")
    if not cid:
        raise RuntimeError("SHOPIFY_CLIENT_ID is missing from .env")
    if not secret:
        raise RuntimeError("SHOPIFY_CLIENT_SECRET is missing from .env")

    r = requests.post(
        f"https://{shop}/admin/oauth/access_token",
        data={
            "grant_type": "client_credentials",
            "client_id": cid,
            "client_secret": secret,
        },
        timeout=30,
    )

    if r.status_code != 200:
        raise RuntimeError(
            f"Shopify authentication failed: HTTP {r.status_code} — {r.text}"
        )

    token = r.json().get("access_token")
    if not token:
        raise RuntimeError("Shopify did not return an access token.")

    return token


def _gql(query, variables, token):
    r = requests.post(
        f"https://{_shop()}/admin/api/{SHOPIFY_API_VERSION}/graphql.json",
        headers={
            "Content-Type": "application/json",
            "X-Shopify-Access-Token": token,
        },
        json={"query": query, "variables": variables},
        timeout=90,
    )

    if r.status_code != 200:
        raise RuntimeError(
            f"Shopify GraphQL HTTP {r.status_code}: {r.text}"
        )

    body = r.json()

    if body.get("errors"):
        raise RuntimeError(
            f"Shopify GraphQL errors: {body['errors']}"
        )

    return body.get("data", {})


def _upload(path, token):
    path = Path(path)

    if not path.exists():
        raise RuntimeError(f"Photo file not found: {path}")

    mime, _ = mimetypes.guess_type(path.name)
    if not mime or not mime.startswith("image/"):
        raise RuntimeError(
            f"Unsupported image type: {path.name} ({mime})"
        )

    query = """
    mutation stagedUploadsCreate($input: [StagedUploadInput!]!) {
      stagedUploadsCreate(input: $input) {
        stagedTargets {
          url
          resourceUrl
          parameters { name value }
        }
        userErrors { field message }
      }
    }
    """

    data = _gql(
        query,
        {
            "input": [{
                "filename": path.name,
                "mimeType": mime,
                "httpMethod": "POST",
                "resource": "PRODUCT_IMAGE",
            }]
        },
        token,
    )

    result = data["stagedUploadsCreate"]

    if result["userErrors"]:
        raise RuntimeError(
            f"Shopify staged upload error: {result['userErrors']}"
        )

    targets = result["stagedTargets"]
    if not targets:
        raise RuntimeError("Shopify returned no staged upload target.")

    target = targets[0]
    form = {
        p["name"]: p["value"]
        for p in target["parameters"]
    }

    with path.open("rb") as fh:
        r = requests.post(
            target["url"],
            data=form,
            files={"file": (path.name, fh, mime)},
            timeout=180,
        )

    if r.status_code not in (200, 201, 204):
        raise RuntimeError(
            f"Shopify image upload failed: HTTP {r.status_code} — {r.text}"
        )

    return target["resourceUrl"]


def _brand(value):
    value = str(value or "").strip()
    if not value or value.lower() in {
        "unknown", "unbranded", "no brand", "none", "n/a",
        "na", "unidentified", "not identifiable"
    }:
        return "Other"
    return value


def _tags(listing):
    raw = listing.get("hashtags", [])
    if isinstance(raw, str):
        raw = raw.split()

    result = []
    for value in raw or []:
        value = str(value).strip().lstrip("#")
        if value and value.lower() not in {x.lower() for x in result}:
            result.append(value)

    return result


def _description(value):
    parts = [
        p.strip()
        for p in str(value or "").splitlines()
        if p.strip()
    ]
    return "".join(
        f"<p>{html.escape(p)}</p>"
        for p in parts
    )


def _listing_category_id(listing):
    """
    Use the exact Shopify taxonomy GID selected by the AI/app for THIS item.

    Never fall back to a global/root category here. The listing's
    category_gid is the source of truth so Shorts, Jeans, Dresses, etc.
    cannot accidentally become Clothing Tops.
    """
    category_id = str(
        listing.get("category_gid", "")
    ).strip()

    if not category_id:
        raise RuntimeError(
            "This listing is missing category_gid. "
            "The AI/app must select a Shopify taxonomy category before "
            "the item can be uploaded."
        )

    if not category_id.startswith(
        "gid://shopify/TaxonomyCategory/"
    ):
        raise RuntimeError(
            "Invalid Shopify category GID: "
            f"{category_id}"
        )

    return category_id


def _clean_option(value):
    value = str(value or "").strip()
    return value


def _product_options(listing):
    """
    Shopify supports productOptions on productCreate.
    For this one-item workflow we create Size and Color options
    with the approved values. The initial variant is generated
    from those option values by Shopify.
    """
    options = []

    size = _clean_option(
        listing.get("size")
    )

    color = _clean_option(
        listing.get("color")
    )

    if size:
        options.append({
            "name": "Size",
            "values": [{
                "name": size
            }]
        })

    if color:
        options.append({
            "name": "Color",
            "values": [{
                "name": color
            }]
        })

    return options[:3]


def _structured_tags(listing):
    """
    Add useful structured values as searchable Shopify tags while
    we keep the actual category-specific Shopify attributes as a
    separate next mapping layer.
    """
    values = []

    def add(value):
        value = str(value or "").strip()

        if (
            value
            and value.lower() not in {
                item.lower()
                for item in values
            }
        ):
            values.append(value)

    add(listing.get("type"))
    add(listing.get("color"))
    add(listing.get("pattern"))
    add(listing.get("neckline"))
    add(listing.get("sleeve_length_type"))
    add(listing.get("sleeve_style"))
    add(listing.get("top_length_type"))
    add(listing.get("fabric"))
    add(listing.get("stretch_level"))
    add(listing.get("top_fit"))

    features = listing.get(
        "clothing_features",
        []
    )

    if isinstance(features, str):
        features = [features]

    for feature in features or []:
        add(feature)

    return values


def create_shopify_draft(listing, photos, token=None):
    """token: pass an already-fetched access token to skip the OAuth
    round trip this call would otherwise make on its own — worth doing
    when uploading a whole batch of drafts back to back, so the caller
    fetches one token up front and reuses it instead of paying for a
    fresh OAuth POST per item."""
    if not isinstance(listing, dict):
        raise RuntimeError("Listing data is invalid.")

    photos = [Path(p) for p in (photos or []) if p]
    if not photos:
        raise RuntimeError("This listing has no photos.")

    token = token or _token()

    media = []
    for photo in photos:
        resource_url = _upload(photo, token)
        media.append({
            "originalSource": resource_url,
            "alt": str(
                listing.get("title", "Clothing item")
            ),
            "mediaContentType": "IMAGE",
        })

    title = str(
        listing.get("title", "Untitled Clothing Item")
    ).strip() or "Untitled Clothing Item"

    category_id = _listing_category_id(listing)

    product_type = str(
        listing.get("type")
        or listing.get("garment_type")
        or "Other"
    ).strip() or "Other"

    tags = _tags(listing)

    for value in _structured_tags(listing):
        if value.lower() not in {
            tag.lower()
            for tag in tags
        }:
            tags.append(value)

    create_query = """
    mutation CreateDraft(
      $product: ProductCreateInput!,
      $media: [CreateMediaInput!]
    ) {
      productCreate(product: $product, media: $media) {
        product {
          id
          title
          handle
          status
          vendor
          productType
          category {
            id
            name
            fullName
          }
          tags
          options {
            id
            name
            position
            optionValues {
              id
              name
              hasVariants
            }
          }
          variants(first: 5) {
            nodes {
              id
              price
              title
              selectedOptions {
                name
                value
              }
            }
          }
          media(first: 50) {
            nodes {
              id
              mediaContentType
            }
          }
        }
        userErrors {
          field
          message
        }
      }
    }
    """

    product_input = {
        "title": title,
        "descriptionHtml": _description(
            listing.get("description", "")
        ),
        "vendor": _brand(
            listing.get("brand")
        ),
        "productType": product_type,
        "category": category_id,
        "tags": tags,
        "status": "DRAFT",
    }

    product_options = _product_options(listing)

    if product_options:
        product_input["productOptions"] = product_options

    data = _gql(
        create_query,
        {
            "product": product_input,
            "media": media,
        },
        token,
    )

    result = data["productCreate"]

    if result["userErrors"]:
        raise RuntimeError(
            f"Shopify product creation error: {result['userErrors']}"
        )

    product = result.get("product")
    if not product:
        raise RuntimeError(
            "Shopify returned no product after productCreate."
        )

    # Shopify must return the same taxonomy category we requested.
    stored_category = product.get("category") or {}
    stored_category_id = str(
        stored_category.get("id", "")
    ).strip()

    if stored_category_id != category_id:
        raise RuntimeError(
            "SHOPIFY CATEGORY MISMATCH. "
            f"Requested: {category_id} | "
            f"Shopify stored: {stored_category_id or 'NONE'} | "
            f"Shopify name: "
            f"{stored_category.get('fullName') or stored_category.get('name') or 'NONE'}"
        )

    print(
        "✓ Shopify category verified: "
        f"{stored_category.get('fullName') or stored_category.get('name')}"
    )

    variants = product.get("variants", {}).get("nodes", [])
    price_raw = listing.get("suggested_price", 0)

    try:
        price = float(price_raw or 0)
    except (TypeError, ValueError):
        price = 0.0

    price_warning = None

    if variants and price > 0:
        update_query = """
        mutation UpdateVariant(
          $productId: ID!,
          $variants: [ProductVariantsBulkInput!]!
        ) {
          productVariantsBulkUpdate(
            productId: $productId,
            variants: $variants
          ) {
            product { id }
            productVariants { id price }
            userErrors { field message }
          }
        }
        """

        try:
            update_data = _gql(
                update_query,
                {
                    "productId": product["id"],
                    "variants": [{
                        "id": variants[0]["id"],
                        "price": f"{price:.2f}",
                    }],
                },
                token,
            )

            errors = update_data[
                "productVariantsBulkUpdate"
            ]["userErrors"]

            if errors:
                price_warning = str(errors)

        except Exception as error:
            price_warning = str(error)

    return {
        "product_id": product["id"],
        "title": product["title"],
        "handle": product.get("handle"),
        "status": product.get("status", "DRAFT"),
        "category": (
            product.get("category", {})
            if product.get("category")
            else None
        ),
        "product_type": product.get("productType"),
        "vendor": product.get("vendor"),
        "tags": product.get("tags", []),
        "options": product.get("options", []),
        "variants": variants,
        "photos_attached": len(
            product.get("media", {}).get("nodes", [])
        ),
        "price": f"{price:.2f}",
        "price_warning": price_warning,
    }
