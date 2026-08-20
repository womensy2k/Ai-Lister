import os
import mimetypes
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

SHOP = os.getenv("SHOPIFY_SHOP")
CLIENT_ID = os.getenv("SHOPIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SHOPIFY_CLIENT_SECRET")

if not SHOP or not CLIENT_ID or not CLIENT_SECRET:
    raise RuntimeError(
        "Missing SHOPIFY_SHOP, SHOPIFY_CLIENT_ID, or "
        "SHOPIFY_CLIENT_SECRET in .env"
    )

SHOP = (
    SHOP.replace("https://", "")
    .replace("http://", "")
    .rstrip("/")
)

if not SHOP.endswith(".myshopify.com"):
    SHOP += ".myshopify.com"

API_URL = (
    f"https://{SHOP}/admin/api/2026-07/graphql.json"
)


def graphql(query, variables, token):
    response = requests.post(
        API_URL,
        headers={
            "Content-Type": "application/json",
            "X-Shopify-Access-Token": token,
        },
        json={
            "query": query,
            "variables": variables,
        },
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Shopify HTTP {response.status_code}: "
            f"{response.text}"
        )

    data = response.json()

    if data.get("errors"):
        raise RuntimeError(
            f"Shopify GraphQL error: {data['errors']}"
        )

    return data["data"]


# ------------------------------------------------------------
# Choose ONE real image from your computer.
# ------------------------------------------------------------

try:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    image_path = filedialog.askopenfilename(
        title="Choose ONE clothing photo for Shopify test",
        filetypes=[
            (
                "Image files",
                "*.jpg *.jpeg *.png *.webp"
            ),
            ("All files", "*.*"),
        ],
    )

    root.destroy()

except Exception as error:
    raise RuntimeError(
        "Could not open the image picker. "
        f"Error: {error}"
    )

if not image_path:
    print("No image selected. Test cancelled.")
    raise SystemExit(0)

image_path = Path(image_path)

if not image_path.exists():
    raise RuntimeError(
        f"Image does not exist: {image_path}"
    )

mime_type, _ = mimetypes.guess_type(
    image_path.name
)

if not mime_type or not mime_type.startswith("image/"):
    raise RuntimeError(
        f"Unsupported image type: {mime_type}"
    )

print()
print("==============================================")
print("SHOPIFY REAL IMAGE TEST")
print("==============================================")
print(f"Selected: {image_path.name}")
print(f"Type: {mime_type}")
print(f"Size: {image_path.stat().st_size:,} bytes")
print()


# ------------------------------------------------------------
# 1. Get Shopify Admin API token.
# ------------------------------------------------------------

print("1/4  Authenticating with Shopify...")

token_response = requests.post(
    f"https://{SHOP}/admin/oauth/access_token",
    headers={
        "Content-Type":
            "application/x-www-form-urlencoded"
    },
    data={
        "grant_type":
            "client_credentials",

        "client_id":
            CLIENT_ID,

        "client_secret":
            CLIENT_SECRET,
    },
    timeout=30,
)

if token_response.status_code != 200:
    raise RuntimeError(
        "Shopify authentication failed: "
        + token_response.text
    )

token = token_response.json().get(
    "access_token"
)

if not token:
    raise RuntimeError(
        "Shopify did not return an access token."
    )

print("    ✓ Authentication successful")


# ------------------------------------------------------------
# 2. Ask Shopify for a staged upload target.
# ------------------------------------------------------------

print("2/4  Preparing Shopify image upload...")

staged_query = """
mutation stagedUploadsCreate(
    $input: [StagedUploadInput!]!
) {
    stagedUploadsCreate(
        input: $input
    ) {
        stagedTargets {
            url
            resourceUrl
            parameters {
                name
                value
            }
        }

        userErrors {
            field
            message
        }
    }
}
"""

staged_data = graphql(
    staged_query,
    {
        "input": [
            {
                "filename":
                    image_path.name,

                "mimeType":
                    mime_type,

                "httpMethod":
                    "POST",

                "resource":
                    "PRODUCT_IMAGE",
            }
        ]
    },
    token,
)

payload = staged_data[
    "stagedUploadsCreate"
]

if payload["userErrors"]:
    raise RuntimeError(
        "Shopify staged upload error: "
        + str(
            payload["userErrors"]
        )
    )

target = payload[
    "stagedTargets"
][0]

print("    ✓ Shopify upload target created")


# ------------------------------------------------------------
# 3. Upload the actual local image to Shopify's staged target.
# ------------------------------------------------------------

print("3/4  Uploading your real image to Shopify...")

upload_data = {}

for parameter in target["parameters"]:
    upload_data[
        parameter["name"]
    ] = parameter["value"]

with image_path.open(
    "rb"
) as image_file:

    upload_response = requests.post(
        target["url"],
        data=upload_data,
        files={
            "file": (
                image_path.name,
                image_file,
                mime_type,
            )
        },
        timeout=120,
    )

if upload_response.status_code not in (
    200,
    201,
    204,
):

    raise RuntimeError(
        "Shopify staged image upload failed: "
        f"HTTP {upload_response.status_code}\n"
        + upload_response.text
    )

print("    ✓ Image uploaded to Shopify staging")


# ------------------------------------------------------------
# 4. Create ONE DRAFT product and attach the uploaded image.
# ------------------------------------------------------------

print("4/4  Creating one Shopify Draft with the image...")

product_query = """
mutation CreateProductWithImage(
    $product: ProductCreateInput!,
    $media: [CreateMediaInput!]
) {

    productCreate(
        product: $product,
        media: $media
    ) {

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

            media(first: 10) {
                nodes {
                    alt
                    mediaContentType
                    ... on MediaImage {
                        image {
                            url
                        }
                    }
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

# ------------------------------------------------------------
# EXACT SHOPIFY TAXONOMY CATEGORY
# ------------------------------------------------------------
# The listing/app should provide this exact Shopify Standard
# Product Taxonomy GID. Example:
#
#   Shorts -> gid://shopify/TaxonomyCategory/aa-1-14
#
# For the real app, pass listing["category_gid"] here instead of
# letting Shopify guess the category.
#
# You can also set SHOPIFY_CATEGORY_GID in .env for this test.
# ------------------------------------------------------------
CATEGORY_GID = os.getenv("SHOPIFY_CATEGORY_GID", "").strip()

if not CATEGORY_GID:
    raise RuntimeError(
        "Missing SHOPIFY_CATEGORY_GID in .env. "
        "Set it to the exact Shopify taxonomy GID from your listing, "
        "for example gid://shopify/TaxonomyCategory/aa-1-14 for Shorts."
    )

if not CATEGORY_GID.startswith("gid://shopify/TaxonomyCategory/"):
    raise RuntimeError(
        "SHOPIFY_CATEGORY_GID must be a Shopify TaxonomyCategory GID, "
        f"got: {CATEGORY_GID}"
    )

product_input = {
    "title":
        "TEST IMAGE - Depop AI",

    "descriptionHtml":
        "<p>Temporary Shopify image integration test.</p>",

    "vendor":
        "Other",

    "productType":
        "Test",

    # IMPORTANT:
    # This is the field that was missing. Shopify's ProductCreateInput
    # accepts category as an ID and will associate the product with
    # that exact Standard Product Taxonomy category.
    "category":
        CATEGORY_GID,

    "tags": [
        "depop-ai-test"
    ],

    "status":
        "DRAFT",
}

media_input = [
    {
        "originalSource":
            target["resourceUrl"],

        "alt":
            "Depop AI Shopify image test",

        "mediaContentType":
            "IMAGE",
    }
]

product_data = graphql(
    product_query,
    {
        "product":
            product_input,

        "media":
            media_input,
    },
    token,
)

result = product_data[
    "productCreate"
]

if result["userErrors"]:
    raise RuntimeError(
        "Shopify product creation error: "
        + str(
            result["userErrors"]
        )
    )

product = result["product"]

print()
print("==============================================")
print("✓ REAL IMAGE TEST SUCCESSFUL")
print("==============================================")
print()
print(
    f"Product: {product['title']}"
)
print(
    f"Status: {product['status']}"
)
print(
    f"Product ID: {product['id']}"
)

stored_category = product.get("category") or {}
print(
    f"Shopify category: {stored_category.get('fullName', 'NOT SET')}"
)
print(
    f"Shopify category GID: {stored_category.get('id', 'NOT SET')}"
)

if stored_category.get("id") != CATEGORY_GID:
    raise RuntimeError(
        "CATEGORY VERIFICATION FAILED: Shopify did not store the "
        "requested taxonomy GID. "
        f"Requested={CATEGORY_GID} "
        f"Stored={stored_category.get('id')}"
    )

print("✓ Exact Shopify taxonomy category verified.")

media_nodes = (
    product
    .get("media", {})
    .get("nodes", [])
)

print(
    f"Images attached: {len(media_nodes)}"
)

if media_nodes:

    image_url = (
        media_nodes[0]
        .get("image", {})
        .get("url")
    )

    if image_url:
        print(
            f"Shopify image URL: {image_url}"
        )

print()
print("✓ The local image was uploaded to Shopify.")
print("✓ A Shopify Draft was created.")
print("✓ The image was attached to the Draft.")
print()
print(
    "Now open Shopify Admin → Products and inspect "
    "the 'TEST IMAGE - Depop AI' Draft."
)
