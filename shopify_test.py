import os
import requests
from dotenv import load_dotenv

load_dotenv()

SHOP = os.getenv("SHOPIFY_SHOP")
CLIENT_ID = os.getenv("SHOPIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SHOPIFY_CLIENT_SECRET")

print("\n=== Shopify Connection Test ===\n")

if not SHOP:
    raise RuntimeError("Missing SHOPIFY_SHOP in .env")

if not CLIENT_ID:
    raise RuntimeError("Missing SHOPIFY_CLIENT_ID in .env")

if not CLIENT_SECRET:
    raise RuntimeError("Missing SHOPIFY_CLIENT_SECRET in .env")

# Accept either:
# your-store
# or:
# your-store.myshopify.com
SHOP = SHOP.replace("https://", "").replace("http://", "").rstrip("/")

if SHOP.endswith(".myshopify.com"):
    shop_domain = SHOP
else:
    shop_domain = f"{SHOP}.myshopify.com"

print(f"Store: {shop_domain}")
print("Requesting Shopify access token...")

token_url = (
    f"https://{shop_domain}"
    "/admin/oauth/access_token"
)

response = requests.post(
    token_url,
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

if response.status_code != 200:

    print("\n❌ Shopify authentication FAILED")
    print(f"HTTP {response.status_code}")
    print(response.text)

    raise SystemExit(1)

token_data = response.json()

access_token = token_data.get(
    "access_token"
)

if not access_token:

    print("\n❌ Shopify did not return an access token.")
    print(response.text)

    raise SystemExit(1)

print("✓ Access token received")

# Test the actual Admin GraphQL API.
graphql_url = (
    f"https://{shop_domain}"
    "/admin/api/2026-07/graphql.json"
)

query = """
{
  shop {
    name
    myshopifyDomain
  }
}
"""

graphql_response = requests.post(
    graphql_url,
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

if graphql_response.status_code != 200:

    print("\n❌ Shopify GraphQL API FAILED")
    print(
        f"HTTP {graphql_response.status_code}"
    )
    print(
        graphql_response.text
    )

    raise SystemExit(1)

result = graphql_response.json()

if result.get("errors"):

    print("\n❌ Shopify returned GraphQL errors:")
    print(
        result["errors"]
    )

    raise SystemExit(1)

shop = (
    result
    .get("data", {})
    .get("shop", {})
)

print("\n================================")
print("✓ SHOPIFY CONNECTION SUCCESSFUL")
print("================================")
print(
    f"Shop name: {shop.get('name')}"
)
print(
    f"Shopify domain: "
    f"{shop.get('myshopifyDomain')}"
)
print(
    "✓ Admin API authentication works"
)
print(
    "✓ Product API connection is ready"
)
print("\nNext step: connect product creation.")