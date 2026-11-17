import { FixedPlayer } from "@/components/players";
export default async function Page({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  return <FixedPlayer id={(await params).id} />;
}
