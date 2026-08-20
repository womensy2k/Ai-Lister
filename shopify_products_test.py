import os
import requests
from dotenv import load_dotenv

load_dotenv()

SHOP = os.getenv("SHOPIFY_SHOP")
CLIENT_ID = os.getenv("SHOPIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SHOPIFY_CLIENT_SECRET")

SHOP = (
    SHOP.replace("https://", "")
    .replace("http://", "")
    .rstrip("/")
)

if not SHOP.endswith(".myshopify.com"):
    SHOP += ".myshopify.com"


# ------------------------------------------------------------
# GET ACCESS TOKEN
# ------------------------------------------------------------

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
    print("❌ Authentication failed")
    print(token_response.text)
    raise SystemExit(1)

access_token = token_response.json()[
    "access_token"
]


# ------------------------------------------------------------
# READ PRODUCTS
# ------------------------------------------------------------

query = """
query {
    products(first: 10) {
        nodes {
            id
            title
            status
            vendor
            productType
            tags
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
            query
    },
    timeout=30,
)

if response.status_code != 200:
    print("❌ Product API failed")
    print(response.text)
    raise SystemExit(1)

data = response.json()

if data.get("errors"):
    print("❌ Shopify returned errors:")
    print(data["errors"])
    raise SystemExit(1)

products = (
    data
    .get("data", {})
    .get("products", {})
    .get("nodes", [])
)

print("\n================================")
print("✓ SHOPIFY PRODUCT API WORKS")
print("================================")

print(
    f"\nFound {len(products)} products "
    "in the first page.\n"
)

for product in products:

    print(
        f"• {product['title']}"
    )

    print(
        f"  Status: {product['status']}"
    )

    print(
        f"  Vendor: {product['vendor']}"
    )

    print(
        f"  Type: {product['productType']}"
    )

    print(
        f"  ID: {product['id']}"
    )

    print()

print("✓ Nothing was created.")
print("✓ Nothing was modified.")
print("✓ read_products is working.")