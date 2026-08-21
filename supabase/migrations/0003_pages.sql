-- Adds the tables needed for the real Dashboard / My Listings /
-- Templates / History / Analytics / Favorites / Settings pages,
-- replacing the "Soon" sidebar placeholders. Same pattern as
-- 0001_init.sql: every table user_id-scoped with RLS so ownership is
-- enforced by Postgres itself.

-- ============================================================
-- TEMPLATES
-- ============================================================

create table public.templates (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    name text not null,
    description text,
    default_condition text,
    default_category text,
    default_category_gid text,
    default_brand text,
    default_hashtags jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

alter table public.templates enable row level security;

create policy "templates_all_own" on public.templates
    for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create index templates_user_id_idx on public.templates(user_id);

-- ============================================================
-- FAVORITES
-- ============================================================

create table public.favorites (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    listing_id uuid not null references public.listings(id) on delete cascade,
    created_at timestamptz not null default now(),
    unique (user_id, listing_id)
);

alter table public.favorites enable row level security;

create policy "favorites_all_own" on public.favorites
    for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create index favorites_user_id_idx on public.favorites(user_id);

-- ============================================================
-- ACTIVITY LOG — item_label is denormalized (kept even if the
-- underlying listing/template is later deleted) so history stays
-- readable instead of showing blank/broken references.
-- ============================================================

create table public.activity_log (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    action text not null check (action in (
        'listing_created', 'listing_updated', 'listing_deleted',
        'template_created', 'template_updated', 'template_deleted'
    )),
    item_type text not null check (item_type in ('listing', 'template')),
    item_id uuid,
    item_label text,
    created_at timestamptz not null default now()
);

alter table public.activity_log enable row level security;

create policy "activity_log_select_own" on public.activity_log
    for select using (auth.uid() = user_id);
create policy "activity_log_insert_own" on public.activity_log
    for insert with check (auth.uid() = user_id);
-- No update/delete policy — history is append-only by design.

create index activity_log_user_id_idx on public.activity_log(user_id, created_at desc);

-- ============================================================
-- PROFILES — listing-preference defaults, applied by the existing
-- Create Listing flow the same way a Template's defaults are.
-- ============================================================

alter table public.profiles
    add column default_condition text,
    add column default_category text,
    add column default_category_gid text,
    add column default_hashtags jsonb not null default '[]'::jsonb;
