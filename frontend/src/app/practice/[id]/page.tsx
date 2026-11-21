import { PracticePlayer } from "@/components/players";
export default async function Page({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  return <PracticePlayer id={(await params).id} />;
}
