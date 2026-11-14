import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TerraSense — Field Decision Support",
  description: "NASA Earth Observation powered management zones and rotation insights",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
