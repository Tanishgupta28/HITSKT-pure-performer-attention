import { redirect } from "next/navigation";

export default async function LegacyPracticeLink({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  redirect("/practice/" + encodeURIComponent((await params).id));
}
