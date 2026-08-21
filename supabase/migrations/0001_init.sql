-- Milestone 1: Auth + Database Foundation
--
-- Run this once in the Supabase project's SQL Editor (or via the
-- Supabase CLI) after creating the project. Every app table carries a
-- user_id column and a Row Level Security policy restricting access to
-- auth.uid() = user_id — ownership is enforced by Postgres itself, not
-- by application code remembering to filter every query.
--
-- Tables here are the full Milestone-1 foundation (matches the schema
-- in the implementation plan) even though this milestone's app code
-- only actively writes to `profiles` and `vintage_corrections` — the
-- rest exist now so later milestones (persistent drafts/batches, photo
-- storage, Shopify OAuth, upload history) just start writing to
-- already-reviewed tables instead of needing their own schema change.

-- ============================================================
-- PROFILES — one row per user, subscription-ready fields inert
-- until billing is actually built.
-- ============================================================

create table public.profiles (
    id uuid primary key references auth.users(id) on delete cascade,
    email text not null,
    plan text not null default 'free',
    subscription_status text,
    subscription_id text,
    billing_period text,
    created_at timestamptz not null default now()
);

alter table public.profiles enable row level security;

create policy "profiles_select_own" on public.profiles
    for select using (auth.uid() = id);
create policy "profiles_update_own" on public.profiles
    for update using (auth.uid() = id);
-- No insert/delete policy for regular users — rows are created only by
-- the handle_new_user() trigger below (running as the table owner,
-- which bypasses RLS), and never deleted directly (cascades from
-- auth.users instead).

-- Auto-create a profile row the moment someone signs up.
create function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
    insert into public.profiles (id, email)
    values (new.id, new.email);
    return new;
end;
$$;

create trigger on_auth_user_created
    after insert on auth.users
    for each row execute procedure public.handle_new_user();

-- ============================================================
-- BATCHES
-- ============================================================

create table public.batches (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    label text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

alter table public.batches enable row level security;

create policy "batches_all_own" on public.batches
    for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create index batches_user_id_idx on public.batches(user_id);

-- ============================================================
-- LISTINGS — one row per confirmed item, mirrors analyze_item()'s
-- full output shape (ai_listing.py) plus lifecycle/status fields.
-- ============================================================

create table public.listings (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    batch_id uuid references public.batches(id) on delete set null,
    item_number integer,
    sku text,

    title text,
    brand text,
    brand_confidence numeric,
    brand_evidence text,
    garment_type text,
    category text,
    category_gid text,
    size text,
    size_confidence numeric,
    size_evidence text,
    color text,
    pattern text,
    style jsonb not null default '[]'::jsonb,
    condition text,
    condition_evidence text,
    description text,
    description_bullets jsonb not null default '[]'::jsonb,
    hashtags jsonb not null default '[]'::jsonb,
    suggested_price numeric,
    cost numeric,

    is_vintage boolean not null default false,
    vintage_classification text,
    vintage_evidence jsonb,
    size_tag_photo_indexes jsonb not null default '[]'::jsonb,

    status text not null default 'draft'
        check (status in ('draft', 'ready', 'listed', 'sold', 'archived')),
    source text
        check (source in ('listing_generator', 'description_generator')),
    shopify_product_id text,

    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

alter table public.listings enable row level security;

create policy "listings_all_own" on public.listings
    for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create index listings_user_id_idx on public.listings(user_id);
create index listings_batch_id_idx on public.listings(batch_id);
create index listings_status_idx on public.listings(user_id, status);

-- ============================================================
-- PHOTOS — one row per photo, references a Supabase Storage path
-- (bucket "listing-photos", key "{user_id}/{listing_id}/{photo_id}.jpg").
-- Wiring uploads to actually write here is a later milestone; the
-- table exists now so that work only needs to start inserting rows.
-- ============================================================

create table public.photos (
    id uuid primary key default gen_random_uuid(),
    listing_id uuid not null references public.listings(id) on delete cascade,
    user_id uuid not null references auth.users(id) on delete cascade,
    storage_path text not null,
    sort_order integer not null default 0,
    is_main boolean not null default false,
    created_at timestamptz not null default now()
);

alter table public.photos enable row level security;

create policy "photos_all_own" on public.photos
    for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create index photos_listing_id_idx on public.photos(listing_id);
create index photos_user_id_idx on public.photos(user_id);

-- ============================================================
-- QA_RESULTS — one row per QA run, kept as history rather than
-- overwritten in place (qa_review.py's run_listing_qa() output shape).
-- ============================================================

create table public.qa_results (
    id uuid primary key default gen_random_uuid(),
    listing_id uuid not null references public.listings(id) on delete cascade,
    user_id uuid not null references auth.users(id) on delete cascade,
    status text not null check (status in ('pass', 'fail')),
    score integer,
    checks jsonb not null default '{}'::jsonb,
    issues jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now()
);

alter table public.qa_results enable row level security;

create policy "qa_results_all_own" on public.qa_results
    for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create index qa_results_listing_id_idx on public.qa_results(listing_id);

-- ============================================================
-- SHOPIFY_CONNECTIONS — one row per user (one connected shop for
-- now). access_token is RLS-protected and must never be selected
-- into any client-readable response — server-side use only.
-- ============================================================

create table public.shopify_connections (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null unique references auth.users(id) on delete cascade,
    shop_domain text not null,
    access_token text not null,
    scope text,
    connected_at timestamptz not null default now()
);

alter table public.shopify_connections enable row level security;

create policy "shopify_connections_all_own" on public.shopify_connections
    for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

-- ============================================================
-- SHOPIFY_UPLOAD_RESULTS — persistent publish history (thrown away
-- every session today).
-- ============================================================

create table public.shopify_upload_results (
    id uuid primary key default gen_random_uuid(),
    listing_id uuid not null references public.listings(id) on delete cascade,
    user_id uuid not null references auth.users(id) on delete cascade,
    shopify_product_id text,
    status text not null check (status in ('success', 'error')),
    error text,
    created_at timestamptz not null default now()
);

alter table public.shopify_upload_results enable row level security;

create policy "shopify_upload_results_all_own" on public.shopify_upload_results
    for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create index shopify_upload_results_listing_id_idx on public.shopify_upload_results(listing_id);

-- ============================================================
-- PRICING_PROFILE — replaces the global pricing_profile.json.
-- One row per (user, bucket_key); upsert on conflict.
-- ============================================================

create table public.pricing_profile (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    bucket_key text not null,
    multiplier numeric not null,
    count integer not null default 1,
    last_manual_price numeric,
    last_market_median numeric,
    style text,
    brand text,
    garment_type text,
    condition text,
    updated_at timestamptz not null default now(),
    unique (user_id, bucket_key)
);

alter table public.pricing_profile enable row level security;

create policy "pricing_profile_all_own" on public.pricing_profile
    for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

-- ============================================================
-- VINTAGE_CORRECTIONS — replaces the global vintage_corrections.json.
-- ============================================================

create table public.vintage_corrections (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    corrected_at timestamptz not null,
    brand text,
    garment_type text,
    bucket_keys jsonb not null default '[]'::jsonb,
    ai_auto_verdict boolean not null,
    from_state boolean not null,
    to_state boolean not null,
    created_at timestamptz not null default now()
);

alter table public.vintage_corrections enable row level security;

create policy "vintage_corrections_all_own" on public.vintage_corrections
    for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create index vintage_corrections_user_id_idx on public.vintage_corrections(user_id);
