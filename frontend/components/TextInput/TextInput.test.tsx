import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { TextInput } from "./TextInput";

describe("TextInput", () => {
  it("calls onChange as the user types", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();

    render(<TextInput value="" onChange={onChange} onSubmit={vi.fn()} onClear={vi.fn()} />);

    await user.type(screen.getByPlaceholderText(/Введіть текст/), "П");

    expect(onChange).toHaveBeenCalledWith("П");
  });

  it("calls onSubmit with the trimmed value when the form is submitted", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();

    render(<TextInput value="  Привіт.  " onChange={vi.fn()} onSubmit={onSubmit} onClear={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "Перекласти" }));

    expect(onSubmit).toHaveBeenCalledWith("Привіт.");
  });

  it("disables the submit button when the value is empty", () => {
    render(<TextInput value="   " onChange={vi.fn()} onSubmit={vi.fn()} onClear={vi.fn()} />);

    expect(screen.getByRole("button", { name: "Перекласти" })).toBeDisabled();
  });

  it("disables the submit button when disabled prop is set", () => {
    render(<TextInput value="Привіт" onChange={vi.fn()} onSubmit={vi.fn()} onClear={vi.fn()} disabled />);

    expect(screen.getByRole("button", { name: "Перекласти" })).toBeDisabled();
  });

  it("Clear button calls onClear and is disabled for an empty value", () => {
    const onClear = vi.fn();
    const { rerender } = render(<TextInput value="" onChange={vi.fn()} onSubmit={vi.fn()} onClear={onClear} />);
    expect(screen.getByRole("button", { name: /Очистити/ })).toBeDisabled();

    rerender(<TextInput value="Привіт" onChange={vi.fn()} onSubmit={vi.fn()} onClear={onClear} />);
    screen.getByRole("button", { name: /Очистити/ }).click();

    expect(onClear).toHaveBeenCalled();
  });

  it("Paste button is hidden when onPaste is not provided, shown and wired when it is", () => {
    const onPaste = vi.fn();
    const { rerender } = render(<TextInput value="" onChange={vi.fn()} onSubmit={vi.fn()} onClear={vi.fn()} />);
    expect(screen.queryByRole("button", { name: /Вставити/ })).not.toBeInTheDocument();

    rerender(<TextInput value="" onChange={vi.fn()} onSubmit={vi.fn()} onClear={vi.fn()} onPaste={onPaste} />);
    screen.getByRole("button", { name: /Вставити/ }).click();

    expect(onPaste).toHaveBeenCalled();
  });

  it("Repeat button is hidden when onRepeat is not provided, shown and wired when it is", () => {
    const onRepeat = vi.fn();
    const { rerender } = render(<TextInput value="Привіт" onChange={vi.fn()} onSubmit={vi.fn()} onClear={vi.fn()} />);
    expect(screen.queryByRole("button", { name: /Повторити переклад/ })).not.toBeInTheDocument();

    rerender(<TextInput value="Привіт" onChange={vi.fn()} onSubmit={vi.fn()} onClear={vi.fn()} onRepeat={onRepeat} />);
    screen.getByRole("button", { name: /Повторити переклад/ }).click();

    expect(onRepeat).toHaveBeenCalled();
  });

  it("Repeat button is disabled for an empty value", () => {
    render(<TextInput value="  " onChange={vi.fn()} onSubmit={vi.fn()} onClear={vi.fn()} onRepeat={vi.fn()} />);

    expect(screen.getByRole("button", { name: /Повторити переклад/ })).toBeDisabled();
  });
});
