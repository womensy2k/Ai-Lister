-- Photo storage bucket + policy, created now alongside the schema even
-- though Milestone 1's app code doesn't upload here yet (Milestone 2
-- wires the upload flow to it) — same reasoning as the unused tables in
-- 0001_init.sql: reviewed and ready rather than added piecemeal later.
--
-- Path convention: listing-photos/{user_id}/{listing_id}/{photo_id}.jpg
-- The policy below restricts every operation to a user's own
-- {user_id}/ prefix, using the same auth.uid() check as the table RLS
-- policies — storage-level enforcement, not just a DB-row check.

insert into storage.buckets (id, name, public)
values ('listing-photos', 'listing-photos', false)
on conflict (id) do nothing;

create policy "listing_photos_all_own"
on storage.objects
for all
using (
    bucket_id = 'listing-photos'
    and (storage.foldername(name))[1] = auth.uid()::text
)
with check (
    bucket_id = 'listing-photos'
    and (storage.foldername(name))[1] = auth.uid()::text
);
