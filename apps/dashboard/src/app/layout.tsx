import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "IntelliCore AI",
  description: "Unified AI observability and lifecycle intelligence",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
