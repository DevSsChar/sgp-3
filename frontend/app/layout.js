import { Space_Grotesk } from "next/font/google";
import "./globals.css";

const spaceGrotesk = Space_Grotesk({
  variable: "--font-space-grotesk",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  display: "swap",
});

export const metadata = {
  title: "Elite AutoML | Data-First Infrastructure",
  description: "Cinematic 3D pipelines for enterprise ML engineers. High-precision automation meets AI infrastructure aesthetics.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body
        className={`${spaceGrotesk.variable} antialiased`}
        style={{ fontFamily: "'Space Grotesk', sans-serif" }}
      >
        {children}
      </body>
    </html>
  );
}
