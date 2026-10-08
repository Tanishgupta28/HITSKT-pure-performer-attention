import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {
  title: "Lumen • Your next learning moment",
  description:
    "A personal learning space powered by knowledge tracing. Understand where you are, and discover what comes next.",
};
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
