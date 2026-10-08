import type { Metadata } from "next";
import localFont from "next/font/local";
import { MotionProvider } from "@/components/motion";
import "./globals.css";
import "./redesign.css";

const bodyFont = localFont({
  src: "./fonts/DMSans-Variable.ttf",
  variable: "--font-body",
  weight: "100 1000",
  display: "swap",
});
const headingFont = localFont({
  src: "./fonts/PlusJakartaSans-Variable.ttf",
  variable: "--font-heading",
  weight: "200 800",
  display: "swap",
});
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
    <html lang="en" className={`${bodyFont.variable} ${headingFont.variable}`}>
      <body>
        <MotionProvider>{children}</MotionProvider>
      </body>
    </html>
  );
}
