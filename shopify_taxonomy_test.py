import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

SHOP = os.getenv("SHOPIFY_SHOP")
CLIENT_ID = os.getenv("SHOPIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SHOPIFY_CLIENT_SECRET")

if not SHOP or not CLIENT_ID or not CLIENT_SECRET:
    raise RuntimeError(
        "Missing SHOPIFY_SHOP, SHOPIFY_CLIENT_ID, "
        "or SHOPIFY_CLIENT_SECRET in .env"
    )

SHOP = (
    SHOP.replace("https://", "")
    .replace("http://", "")
    .rstrip("/")
)

if not SHOP.endswith(".myshopify.com"):
    SHOP += ".myshopify.com"

API_URL = f"https://{SHOP}/admin/api/2026-07/graphql.json"


# ============================================================
# AUTHENTICATE
# ============================================================

print("\nConnecting to Shopify...")

token_response = requests.post(
    f"https://{SHOP}/admin/oauth/access_token",
    headers={
        "Content-Type": "application/x-www-form-urlencoded"
    },
    data={
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    },
    timeout=30,
)

if token_response.status_code != 200:
    raise RuntimeError(
        f"Shopify authentication failed: "
        f"HTTP {token_response.status_code}\n"
        f"{token_response.text}"
    )

TOKEN = token_response.json().get("access_token")

if not TOKEN:
    raise RuntimeError(
        "Shopify did not return an access token."
    )

print("✓ Shopify authentication successful")


# ============================================================
# GRAPHQL HELPER
# ============================================================

def graphql(query, variables=None):

    response = requests.post(
        API_URL,
        headers={
            "Content-Type": "application/json",
            "X-Shopify-Access-Token": TOKEN,
        },
        json={
            "query": query,
            "variables": variables or {},
        },
        timeout=90,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Shopify HTTP {response.status_code}:\n"
            f"{response.text}"
        )

    data = response.json()

    if data.get("errors"):
        raise RuntimeError(
            json.dumps(
                data["errors"],
                indent=2
            )
        )

    return data["data"]


# ============================================================
# SEARCH TAXONOMY
# ============================================================

print("\nSearching Shopify's taxonomy for Jeans...")

search_query = """
query SearchTaxonomy($search: String!) {
    taxonomy {
        categories(
            first: 250
            search: $search
        ) {
            nodes {
                id
                name
                fullName
                level
                isLeaf
                isArchived
                parentId
            }
        }
    }
}
"""

data = graphql(
    search_query,
    {
        "search": "jeans"
    }
)

search_categories = (
    data
    .get("taxonomy", {})
    .get("categories", {})
    .get("nodes", [])
)

search_categories = [
    category
    for category in search_categories
    if not category.get("isArchived")
]

# Only accept a real Jeans category under Apparel & Accessories > Clothing.
categories = [
    category
    for category in search_categories
    if (
        str(category.get("name", "")).strip().lower() == "jeans"
        and "apparel & accessories > clothing"
        in str(category.get("fullName", "")).strip().lower()
    )
]

if not categories:
    # A second exact-ish search catches stores where Shopify's search
    # ranks the parent hierarchy differently.
    data = graphql(
        search_query,
        {
            "search": "clothing jeans"
        }
    )

    search_categories = (
        data
        .get("taxonomy", {})
        .get("categories", {})
        .get("nodes", [])
    )

    search_categories = [
        category
        for category in search_categories
        if not category.get("isArchived")
    ]

    categories = [
        category
        for category in search_categories
        if (
            "jeans"
            in str(category.get("name", "")).strip().lower()
            and "apparel & accessories > clothing"
            in str(category.get("fullName", "")).strip().lower()
        )
    ]

if not categories:
    print("\nShopify returned these Jeans search matches:")
    for category in search_categories:
        print(
            f"- [{category.get('fullName')}] "
            f"{category.get('id')}"
        )

    raise RuntimeError(
        "Could not find the Jeans category in Shopify's "
        "Apparel & Accessories > Clothing taxonomy."
    )

print(
    f"\n✓ Found {len(categories)} Jeans category candidate(s):\n"
)

for number, category in enumerate(
    categories,
    start=1
):
    leaf = "LEAF" if category.get("isLeaf") else "PARENT"

    print(
        f"{number}. [{leaf}] "
        f"{category.get('fullName')}"
    )

    print(
        f"   ID: {category.get('id')}"
    )

print()


print(
    f"\n✓ Found {len(categories)} categories:\n"
)

for number, category in enumerate(
    categories,
    start=1
):
    leaf = "LEAF" if category.get("isLeaf") else "PARENT"

    print(
        f"{number}. [{leaf}] "
        f"{category.get('fullName')}"
    )

    print(
        f"   ID: {category.get('id')}"
    )

print()


# ============================================================
# LET YOU CHOOSE THE CATEGORY
# ============================================================

while True:

    answer = input(
        "Enter the number of the category you want to inspect "
        "(or press Enter for the first leaf): "
    ).strip()

    if not answer:

        leaf_categories = [
            category
            for category in categories
            if category.get("isLeaf")
        ]

        selected = (
            leaf_categories[0]
            if leaf_categories
            else categories[0]
        )

        break

    try:
        number = int(answer)

    except ValueError:
        print("Enter a number.")
        continue

    if 1 <= number <= len(categories):

        selected = categories[number - 1]
        break

    print(
        f"Enter a number from 1 to {len(categories)}."
    )


category_id = selected["id"]

print("\n==============================================")
print("SELECTED SHOPIFY CATEGORY")
print("==============================================")
print(selected["fullName"])
print(f"ID: {category_id}")
print()


# ============================================================
# GET THE ACTUAL CATEGORY NODE BY ID
# ============================================================
#
# This is the part the previous script got wrong.
#
# taxonomy.categories(search: ...) expects a STRING search term.
# We were trying to pass a Shopify ID into that search argument,
# which caused:
#
# variableMismatch: ID! -> search
#
# Instead, TaxonomyCategory implements Node, so we query node(id:)
# and ask for the TaxonomyCategory fields directly.
# ============================================================

category_query = """
query GetTaxonomyCategory($id: ID!) {

    node(id: $id) {

        ... on TaxonomyCategory {

            id
            name
            fullName
            isLeaf
            parentId

            attributes(first: 100) {

                nodes {

                    __typename

                    ... on TaxonomyChoiceListAttribute {

                        id
                        name

                        values(first: 250) {

                            nodes {
                                id
                                name
                            }

                        }
                    }

                    ... on TaxonomyMeasurementAttribute {

                        id
                        name
                    }

                    ... on TaxonomyAttribute {

                        id
                    }
                }
            }
        }
    }
}
"""

category_data = graphql(
    category_query,
    {
        "id": category_id
    }
)

category = category_data.get("node")

if not category:
    raise RuntimeError(
        "Shopify returned no TaxonomyCategory for "
        f"{category_id}"
    )

print(
    f"✓ Retrieved category: {category['fullName']}\n"
)


# ============================================================
# EXTRACT ATTRIBUTES
# ============================================================

result = {
    "category": {
        "id": category["id"],
        "name": category["name"],
        "full_name": category["fullName"],
        "is_leaf": category["isLeaf"],
        "parent_id": category.get("parentId"),
    },
    "attributes": {}
}

attributes = (
    category
    .get("attributes", {})
    .get("nodes", [])
)

print("SHOPIFY JEANS CATEGORY ATTRIBUTES")
print("==============================================")

if not attributes:

    print(
        "No attributes were returned for this category."
    )

else:

    for attribute in attributes:

        name = attribute.get("name")

        if not name:
            continue

        values = (
            attribute
            .get("values", {})
            .get("nodes", [])
        )

        clean_values = [
            {
                "id": value.get("id"),
                "name": value.get("name"),
            }
            for value in values
            if value.get("name")
        ]

        result["attributes"][name] = {
            "id": attribute.get("id"),
            "values": clean_values,
        }

        print()
        print(name)
        print(
            f"Shopify attribute ID: "
            f"{attribute.get('id')}"
        )

        if clean_values:

            print(
                f"Allowed values: "
                f"{len(clean_values)}"
            )

            for value in clean_values:
                print(
                    f"  • {value['name']}"
                )

        else:

            print(
                "No choice-list values."
            )


# ============================================================
# SAVE SOURCE-OF-TRUTH JSON
# ============================================================

output_file = "shopify_taxonomy_options_jeans.json"

with open(
    output_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        result,
        file,
        indent=2,
        ensure_ascii=False
    )


print()
print("==============================================")
print("✓ SHOPIFY TAXONOMY DATA SAVED")
print("==============================================")
print(f"File: {output_file}")
print()
print(
    "This JSON will become the source of truth for "
    "the AI's allowed Shopify values."
)
