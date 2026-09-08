import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AVM CatBoost Dashboard",
  description: "Hemorrhagic presentation prediction dashboard"
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}