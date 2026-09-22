import nextPlugin from "eslint-config-next";

const eslintConfig = [
  ...nextPlugin,
  {
    ignores: ["node_modules/**", ".next/**"],
  },
];

export default eslintConfig;
