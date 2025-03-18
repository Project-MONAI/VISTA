import argparse

import torch


def remove_module_prefix(input_pth, output_pth):
    # Load the checkpoint
    checkpoint = torch.load(input_pth, map_location="cpu")["model"]

    # Modify the state_dict to remove 'module.' prefix
    new_state_dict = {}
    for key, value in checkpoint.items():
        if isinstance(value, dict) and "state_dict" in value:
            # If the checkpoint contains a 'state_dict' key (common in some saved models)
            new_state_dict = {
                k.replace("module.", ""): v for k, v in value["state_dict"].items()
            }
            value["state_dict"] = new_state_dict
            torch.save(value, output_pth)
            print(f"Updated weights saved to {output_pth}")
            return
        elif "module." in key:
            new_state_dict[key.replace("module.", "")] = value
        else:
            new_state_dict[key] = value

    # Save the modified weights
    torch.save(new_state_dict, output_pth)
    print(f"Updated weights saved to {output_pth}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove 'module.' prefix from PyTorch weights"
    )
    parser.add_argument("--input", required=True, help="Path to input .pth file")
    parser.add_argument(
        "--output", required=True, help="Path to save the modified .pth file"
    )
    args = parser.parse_args()

    remove_module_prefix(args.input, args.output)
