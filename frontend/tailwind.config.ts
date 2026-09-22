import type { Config } from "tailwindcss";
import colors from "tailwindcss/colors";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Single accent color for the whole app (buttons, links, active
        // states, focus rings) -- every component reads "brand-*", never a
        // hardcoded Tailwind color, so re-theming stays a one-line change.
        brand: colors.red,
      },
    },
  },
  plugins: [],
};

export default config;
