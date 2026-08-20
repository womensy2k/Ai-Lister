import os
import requests
from dotenv import load_dotenv

load_dotenv()

SHOP = os.getenv("SHOPIFY_SHOP")
CLIENT_ID = os.getenv("SHOPIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SHOPIFY_CLIENT_SECRET")

if not SHOP:
    raise RuntimeError("Missing SHOPIFY_SHOP")

if not CLIENT_ID:
    raise RuntimeError("Missing SHOPIFY_CLIENT_ID")

if not CLIENT_SECRET:
    raise RuntimeError("Missing SHOPIFY_CLIENT_SECRET")

SHOP = (
    SHOP.replace("https://", "")
    .replace("http://", "")
    .rstrip("/")
)

if not SHOP.endswith(".myshopify.com"):
    SHOP += ".myshopify.com"


# ============================================================
# 1. GET ADMIN API ACCESS TOKEN
# ============================================================

print("\nConnecting to Shopify...")

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

    print("\n❌ Authentication failed")
    print(token_response.text)

    raise SystemExit(1)

access_token = token_response.json().get(
    "access_token"
)

if not access_token:

    print("\n❌ Shopify did not return an access token.")
    print(token_response.text)

    raise SystemExit(1)

print("✓ Shopify authentication successful")


# ============================================================
# 2. TEST PRODUCT DATA
# ============================================================
#
# This is intentionally hard-coded for the FIRST test.
# Once this works, we'll replace this with your AI listing data.
#

product = {
    "title":
        "TEST - Y2K Brown American Eagle Top",

    "descriptionHtml":
        """
        <p>
        TEST PRODUCT — Y2K American Eagle top.
        This is a temporary Shopify integration test.
        </p>
        """,

    "vendor":
        "American Eagle",

    "productType":
        "Tops",

    "tags": [
        "y2k",
        "american-eagle",
        "brown",
        "vintage",
        "womens"
    ],

    "status":
        "DRAFT",

    "productOptions": [
        {
            "name":
                "Size",

            "values": [
                {
                    "name":
                        "Small"
                }
            ]
        }
    ]
}


# ============================================================
# 3. CREATE SHOPIFY DRAFT
# ============================================================

mutation = """
mutation CreateTestProduct(
    $product: ProductCreateInput!
) {

    productCreate(
        product: $product
    ) {

        product {
            id
            title
            handle
            status
            vendor
            productType
            tags

            variants(first: 1) {
                nodes {
                    id
                    title
                    price
                    selectedOptions {
                        name
                        value
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


response = requests.post(
    f"https://{SHOP}/admin/api/2026-07/graphql.json",

    headers={
        "Content-Type":
            "application/json",

        "X-Shopify-Access-Token":
            access_token,
    },

    json={
        "query":
            mutation,

        "variables": {
            "product":
                product
        }
    },

    timeout=30,
)


# ============================================================
# 4. CHECK RESPONSE
# ============================================================

if response.status_code != 200:

    print("\n❌ Shopify API request failed")
    print(
        f"HTTP {response.status_code}"
    )
    print(
        response.text
    )

    raise SystemExit(1)


data = response.json()

if data.get("errors"):

    print("\n❌ Shopify GraphQL error")
    print(
        data["errors"]
    )

    raise SystemExit(1)


result = (
    data
    .get("data", {})
    .get("productCreate", {})
)

user_errors = result.get(
    "userErrors",
    []
)

if user_errors:

    print("\n❌ Shopify rejected the product:")

    for error in user_errors:

        print(
            f"- {error.get('message')}"
        )

    raise SystemExit(1)


created = result.get(
    "product"
)

if not created:

    print(
        "\n❌ Shopify did not return a product."
    )

    print(
        response.text
    )

    raise SystemExit(1)


# ============================================================
# SUCCESS
# ============================================================

print("\n")
print("==============================================")
print("✓ SHOPIFY TEST PRODUCT CREATED")
print("==============================================")

print(
    f"\nTitle: "
    f"{created.get('title')}"
)

print(
    f"Status: "
    f"{created.get('status')}"
)

print(
    f"Vendor: "
    f"{created.get('vendor')}"
)

print(
    f"Product Type: "
    f"{created.get('productType')}"
)

print(
    f"Tags: "
    f"{created.get('tags')}"
)

print(
    f"Product ID: "
    f"{created.get('id')}"
)

print(
    f"Handle: "
    f"{created.get('handle')}"
)

variants = (
    created
    .get("variants", {})
    .get("nodes", [])
)

if variants:

    variant = variants[0]

    print(
        f"Variant: "
        f"{variant.get('title')}"
    )

    print(
        f"Price: "
        f"{variant.get('price')}"
    )

print("\n✓ This product is a DRAFT.")
print("✓ It was NOT published to your storefront.")
print("\nGo to Shopify Admin → Products and verify it.")