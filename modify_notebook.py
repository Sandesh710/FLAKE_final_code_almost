#!/usr/bin/env python3
"""
Script to modify the FLake notebook to output .rslt file matching .test format
"""
import json
import sys

def modify_notebook():
    # Load notebook
    with open('FLAKE_DEC5_NEW_TEMPR_ALMOST_WORKING-Copy2.ipynb', 'r') as f:
        nb = json.load(f)

    print(f"Loaded notebook with {len(nb['cells'])} cells")

    # Find critical cells
    flake_interface_cell = None
    main_loop_cell = None

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'def flake_interface(' in source and flake_interface_cell is None:
                flake_interface_cell = i
                print(f"Found flake_interface at cell {i}")
            if 'for k in range(n_steps)' in source and 'output_data' in source:
                main_loop_cell = i
                print(f"Found main loop at cell {i}")

    if flake_interface_cell is None:
        print("ERROR: Could not find flake_interface function")
        return False

    if main_loop_cell is None:
        print("ERROR: Could not find main simulation loop")
        return False

    # Get the flake_interface source
    flake_source = ''.join(nb['cells'][flake_interface_cell]['source'])

    # Check if we need to modify the return statement
    if "'ufr_a_out'" not in flake_source:
        print("Modifying flake_interface to return flux variables...")

        # Find the return statement
        lines = flake_source.split('\n')
        return_idx = None
        for idx, line in enumerate(lines):
            if 'return {' in line or "return dict(" in line:
                return_idx = idx
                break

        if return_idx is None:
            print("ERROR: Could not find return statement in flake_interface")
            return False

        # Find the closing brace
        brace_count = 0
        end_idx = return_idx
        for idx in range(return_idx, len(lines)):
            brace_count += lines[idx].count('{') - lines[idx].count('}')
            if brace_count == 0 and '}' in lines[idx]:
                end_idx = idx
                break

        # Insert flux variables before the closing brace
        new_lines = lines[:end_idx] + [
            "        # Flux variables for output",
            "        'ufr_a_out': u_star_a_sf,  # Friction velocity in air",
            "        'ufr_w_out': u_star_w_flk,  # Friction velocity in water",
            "        'Wconv_out': w_star_sfc_flk,  # Convective velocity",
            "        'Q_momentum_out': Q_momentum_val,  # Momentum flux",
            "        'Q_sensible_out': Q_sensible_val,  # Sensible heat flux",
            "        'Q_latent_out': Q_latent_val,  # Latent heat flux",
            "        'I_w_out': I_w_flk,  # Solar radiation in water",
            "        'Q_lwa_out': Q_atm_lw_in,  # Longwave from atmosphere (input)",
            "        'Q_lww_out': Q_lww_val,  # Longwave from water surface",
            "        'Q_w_out': Q_w_flk,  # Total heat flux to water",
            "        'Q_bot_out': Q_bot_val  # Bottom heat flux",
        ] + lines[end_idx:]

        # We also need to capture these values before returning
        # Find where to insert the value captures (before return statement)
        capture_code = [
            "    # Capture flux values for output",
            "    Q_momentum_val = Q_momentum if 'Q_momentum' in dir() else 0.0",
            "    Q_sensible_val = Q_sensible if 'Q_sensible' in dir() else 0.0",
            "    Q_latent_val = Q_latent if 'Q_latent' in dir() else 0.0",
            "    Q_lww_val = SfcFlx_lwradwsfc(T_sfc_n) if T_sfc_n > 0 else 0.0",
            "    Q_bot_val = Q_bot_flk if 'Q_bot_flk' in dir() else 0.0",
            "    ",
        ]

        new_lines = new_lines[:return_idx] + capture_code + new_lines[return_idx:]

        # Update the cell
        nb['cells'][flake_interface_cell]['source'] = '\n'.join(new_lines)
        print("✓ Modified flake_interface return statement")
    else:
        print("flake_interface already modified")

    # Save modified notebook
    output_file = 'FLAKE_DEC5_NEW_TEMPR_ALMOST_WORKING-Copy2.ipynb'
    with open(output_file, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"✓ Saved modified notebook to {output_file}")
    return True

if __name__ == '__main__':
    success = modify_notebook()
    sys.exit(0 if success else 1)
