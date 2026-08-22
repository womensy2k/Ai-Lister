-- Web Push notification support for the installed home-screen PWA.
--
-- The subscribing page (docs/notifications.html) runs on a DIFFERENT
-- origin (GitHub Pages) than this app (Streamlit Cloud) and has no
-- Supabase auth session at all — iOS requires the notification
-- permission prompt and service-worker registration to happen from
-- the same origin/scope as the installed PWA's manifest, which lives
-- on GitHub Pages, not here. So it can't just INSERT into
-- push_subscriptions under normal RLS (auth.uid() is null there).
--
-- Instead: the logged-in Settings page (real Supabase session) mints
-- a short-lived, single-use pairing token tied to its own user_id.
-- The user opens notifications.html with that token in the URL; that
-- page calls the consume_push_pairing_token() RPC below, which is the
-- ONLY way a subscription row ever gets created — it runs
-- SECURITY DEFINER specifically so it can resolve "this token proves
-- you're user X" and insert on their behalf without a real session,
-- while everything else about these tables stays RLS-locked to
-- auth.uid() like every other table in this app.

-- ============================================================
-- PUSH SUBSCRIPTIONS
-- ============================================================

create table public.push_subscriptions (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    endpoint text not null,
    p256dh_key text not null,
    auth_key text not null,
    user_agent text,
    created_at timestamptz not null default now(),
    unique (user_id, endpoint)
);

alter table public.push_subscriptions enable row level security;

-- Authenticated users can see/remove their own subscriptions (e.g. a
-- "notifications enabled on 2 devices" list + "remove this device" in
-- Settings) — but NOT insert directly; new subscriptions only ever
-- come from the pairing-token RPC below.
create policy "push_subscriptions_select_own" on public.push_subscriptions
    for select using (auth.uid() = user_id);
create policy "push_subscriptions_delete_own" on public.push_subscriptions
    for delete using (auth.uid() = user_id);

create index push_subscriptions_user_id_idx on public.push_subscriptions(user_id);

-- ============================================================
-- PAIRING TOKENS — minted by the logged-in Settings page, consumed
-- once (or left to expire) by the unauthenticated notifications.html
-- page via the RPC function below.
-- ============================================================

create table public.push_pairing_tokens (
    token text primary key,
    user_id uuid not null references auth.users(id) on delete cascade,
    expires_at timestamptz not null,
    consumed_at timestamptz
);

alter table public.push_pairing_tokens enable row level security;

create policy "push_pairing_tokens_all_own" on public.push_pairing_tokens
    for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create index push_pairing_tokens_user_id_idx on public.push_pairing_tokens(user_id);

-- ============================================================
-- CONSUME-TOKEN RPC — the one deliberate, narrow hole in RLS above.
-- SECURITY DEFINER so it can look up the token's owner and insert the
-- subscription on their behalf even though the CALLER (an anonymous
-- GitHub Pages page) has no auth.uid() of its own. search_path is
-- pinned and every name fully schema-qualified so this can't be tricked
-- by a hostile search_path — standard hardening for SECURITY DEFINER
-- functions exposed to the anon role.
-- ============================================================

create or replace function public.consume_push_pairing_token(
    p_token text,
    p_endpoint text,
    p_p256dh_key text,
    p_auth_key text,
    p_user_agent text default null
)
returns boolean
language plpgsql
security definer
set search_path = ''
as $$
declare
    v_user_id uuid;
begin
    select user_id into v_user_id
    from public.push_pairing_tokens
    where token = p_token
      and expires_at > now()
      and consumed_at is null;

    if v_user_id is null then
        return false;
    end if;

    update public.push_pairing_tokens
    set consumed_at = now()
    where token = p_token;

    insert into public.push_subscriptions (user_id, endpoint, p256dh_key, auth_key, user_agent)
    values (v_user_id, p_endpoint, p_p256dh_key, p_auth_key, p_user_agent)
    on conflict (user_id, endpoint) do update
        set p256dh_key = excluded.p256dh_key,
            auth_key = excluded.auth_key,
            user_agent = excluded.user_agent;

    return true;
end;
$$;

-- The anon key is all notifications.html has — grant it (and
-- authenticated, harmless either way) execute on this one function
-- only. No table grants are needed/given beyond what RLS above
-- already allows.
grant execute on function public.consume_push_pairing_token(text, text, text, text, text) to anon, authenticated;
