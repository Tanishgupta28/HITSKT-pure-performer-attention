import { JunyiPlayer } from "@/components/junyi";

export default async function Page({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <JunyiPlayer id={id} />;
}
