import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { CorrectionPopover } from "./CorrectionPopover";

const OPTIONS = [
  { key: "WANT", label: "хотіти", confidence: 0.6 },
  { key: "HAVE", label: "мати", confidence: 0.35 },
  { key: "WATER", label: "вода", confidence: 0.05 },
];

describe("CorrectionPopover", () => {
  it("starts closed, showing only the trigger label", () => {
    render(
      <CorrectionPopover
        triggerLabel="хотіти"
        currentKey="WANT"
        options={OPTIONS}
        ariaLabel="Виправити слово «хотіти»"
        onSelect={() => {}}
      />,
    );

    expect(screen.getByRole("button", { name: "Виправити слово «хотіти»" })).toBeInTheDocument();
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  });

  it("opens on click and lists every real candidate with its real confidence, ranked pick labeled", async () => {
    const user = userEvent.setup();
    render(
      <CorrectionPopover
        triggerLabel="хотіти"
        currentKey="WANT"
        options={OPTIONS}
        ariaLabel="Виправити слово «хотіти»"
        onSelect={() => {}}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Виправити слово «хотіти»" }));

    expect(screen.getByRole("option", { name: "хотіти 60%" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "мати 35%" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "вода 5%" })).toBeInTheDocument();
  });

  it("marks the currently selected candidate as aria-selected", async () => {
    const user = userEvent.setup();
    render(
      <CorrectionPopover
        triggerLabel="хотіти"
        currentKey="WANT"
        options={OPTIONS}
        ariaLabel="Виправити слово «хотіти»"
        onSelect={() => {}}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Виправити слово «хотіти»" }));

    expect(screen.getByRole("option", { name: "хотіти 60%" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("option", { name: "мати 35%" })).toHaveAttribute("aria-selected", "false");
  });

  it("calls onSelect with the picked candidate's key and closes the list", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(
      <CorrectionPopover
        triggerLabel="хотіти"
        currentKey="WANT"
        options={OPTIONS}
        ariaLabel="Виправити слово «хотіти»"
        onSelect={onSelect}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Виправити слово «хотіти»" }));
    await user.click(screen.getByRole("option", { name: "мати 35%" }));

    expect(onSelect).toHaveBeenCalledWith("HAVE");
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  });

  it("closes on Escape without calling onSelect", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(
      <CorrectionPopover
        triggerLabel="хотіти"
        currentKey="WANT"
        options={OPTIONS}
        ariaLabel="Виправити слово «хотіти»"
        onSelect={onSelect}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Виправити слово «хотіти»" }));
    expect(screen.getByRole("listbox")).toBeInTheDocument();

    await user.keyboard("{Escape}");

    await waitFor(() => expect(screen.queryByRole("listbox")).not.toBeInTheDocument());
    expect(onSelect).not.toHaveBeenCalled();
  });

  it("closes when clicking outside the popover", async () => {
    const user = userEvent.setup();
    render(
      <div>
        <button type="button">outside</button>
        <CorrectionPopover
          triggerLabel="хотіти"
          currentKey="WANT"
          options={OPTIONS}
          ariaLabel="Виправити слово «хотіти»"
          onSelect={() => {}}
        />
      </div>,
    );

    await user.click(screen.getByRole("button", { name: "Виправити слово «хотіти»" }));
    expect(screen.getByRole("listbox")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "outside" }));

    await waitFor(() => expect(screen.queryByRole("listbox")).not.toBeInTheDocument());
  });

  it("calls onOpenChange(true) when opened and onOpenChange(false) when a selection closes it", async () => {
    const user = userEvent.setup();
    const onOpenChange = vi.fn();
    render(
      <CorrectionPopover
        triggerLabel="хотіти"
        currentKey="WANT"
        options={OPTIONS}
        ariaLabel="Виправити слово «хотіти»"
        onSelect={() => {}}
        onOpenChange={onOpenChange}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Виправити слово «хотіти»" }));
    expect(onOpenChange).toHaveBeenLastCalledWith(true);

    await user.click(screen.getByRole("option", { name: "мати 35%" }));
    expect(onOpenChange).toHaveBeenLastCalledWith(false);
  });

  it("calls onOpenChange(false) when closed via Escape or an outside click", async () => {
    const user = userEvent.setup();
    const onOpenChange = vi.fn();
    render(
      <div>
        <button type="button">outside</button>
        <CorrectionPopover
          triggerLabel="хотіти"
          currentKey="WANT"
          options={OPTIONS}
          ariaLabel="Виправити слово «хотіти»"
          onSelect={() => {}}
          onOpenChange={onOpenChange}
        />
      </div>,
    );

    await user.click(screen.getByRole("button", { name: "Виправити слово «хотіти»" }));
    expect(onOpenChange).toHaveBeenLastCalledWith(true);

    await user.keyboard("{Escape}");
    await waitFor(() => expect(onOpenChange).toHaveBeenLastCalledWith(false));

    await user.click(screen.getByRole("button", { name: "Виправити слово «хотіти»" }));
    expect(onOpenChange).toHaveBeenLastCalledWith(true);

    await user.click(screen.getByRole("button", { name: "outside" }));
    await waitFor(() => expect(onOpenChange).toHaveBeenLastCalledWith(false));
  });

  it("toggles closed when the trigger is clicked again", async () => {
    const user = userEvent.setup();
    render(
      <CorrectionPopover
        triggerLabel="хотіти"
        currentKey="WANT"
        options={OPTIONS}
        ariaLabel="Виправити слово «хотіти»"
        onSelect={() => {}}
      />,
    );

    const trigger = screen.getByRole("button", { name: "Виправити слово «хотіти»" });
    await user.click(trigger);
    expect(screen.getByRole("listbox")).toBeInTheDocument();

    await user.click(trigger);
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  });
});
