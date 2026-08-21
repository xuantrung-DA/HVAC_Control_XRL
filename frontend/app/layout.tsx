import "@fontsource-variable/manrope";
import "@fontsource-variable/space-grotesk";
import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "XRL-HVAC · Explainable Building Intelligence",
  description: "Explainable reinforcement learning for smart-building HVAC control.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
