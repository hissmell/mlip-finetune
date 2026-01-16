"""VASP output to extxyz converter.

Converts VASP calculation results (OUTCAR, vasprun.xml) to extxyz format
suitable for MLIP training.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import warnings

import numpy as np
from ase import Atoms
from ase.io import write as ase_write
from tqdm import tqdm

logger = logging.getLogger(__name__)


def find_vasp_directories(root_dir: str) -> List[Path]:
    """
    Find all directories containing VASP output files.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        List of paths to directories containing OUTCAR or vasprun.xml
    """
    root = Path(root_dir)
    vasp_dirs = []
    
    for dirpath, dirnames, filenames in os.walk(root):
        # Check for VASP output files
        if 'OUTCAR' in filenames or 'vasprun.xml' in filenames:
            vasp_dirs.append(Path(dirpath))
    
    return sorted(vasp_dirs)


def read_outcar(outcar_path: Path) -> Optional[List[Atoms]]:
    """
    Try to read structures from OUTCAR using ASE.
    
    Args:
        outcar_path: Path to OUTCAR file
        
    Returns:
        List of Atoms objects if successful, None otherwise
    """
    try:
        from ase.io import read
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atoms_list = read(str(outcar_path), index=':', format='vasp-out')
        
        # Verify that we have forces and energies
        if not atoms_list:
            return None
        
        # Check first and last frame for forces/energy
        for atoms in [atoms_list[0], atoms_list[-1]]:
            try:
                _ = atoms.get_potential_energy()
                forces = atoms.get_forces()
                if forces is None or len(forces) == 0:
                    logger.debug(f"No forces in {outcar_path}")
                    return None
            except Exception as e:
                logger.debug(f"Failed to get energy/forces from {outcar_path}: {e}")
                return None
        
        # Extract energy/forces from calculator and store in info/arrays
        # ASE stores them in the calculator, not in info/arrays
        result = []
        for atoms in atoms_list:
            new_atoms = atoms.copy()
            if atoms.calc is not None:
                try:
                    new_atoms.info['energy'] = atoms.get_potential_energy()
                except:
                    pass
                try:
                    new_atoms.arrays['forces'] = atoms.get_forces()
                except:
                    pass
                try:
                    new_atoms.info['stress'] = atoms.get_stress()
                except:
                    pass
            result.append(new_atoms)
        
        return result
        
    except Exception as e:
        logger.debug(f"Failed to read OUTCAR {outcar_path}: {e}")
        return None


def read_xdatcar(xdatcar_path: Path) -> Optional[List[Atoms]]:
    """
    Read structures from XDATCAR.
    
    Args:
        xdatcar_path: Path to XDATCAR file
        
    Returns:
        List of Atoms objects (without energy/forces)
    """
    try:
        from ase.io import read
        
        atoms_list = read(str(xdatcar_path), index=':', format='vasp-xdatcar')
        return atoms_list if atoms_list else None
        
    except Exception as e:
        logger.debug(f"Failed to read XDATCAR {xdatcar_path}: {e}")
        return None


def read_vasprun_xml(vasprun_path: Path) -> Optional[List[Dict[str, Any]]]:
    """
    Read energies and forces from vasprun.xml.
    
    Args:
        vasprun_path: Path to vasprun.xml file
        
    Returns:
        List of dicts with 'energy', 'forces', 'stress' for each ionic step
    """
    try:
        from xml.etree import ElementTree as ET
        
        tree = ET.parse(str(vasprun_path))
        root = tree.getroot()
        
        results = []
        
        # Find all calculation blocks (ionic steps)
        for calculation in root.findall('.//calculation'):
            data = {}
            
            # Get energy
            energy_elem = calculation.find('.//energy/i[@name="e_fr_energy"]')
            if energy_elem is None:
                energy_elem = calculation.find('.//energy/i[@name="e_0_energy"]')
            
            if energy_elem is not None:
                data['energy'] = float(energy_elem.text)
            
            # Get forces
            forces_block = calculation.find('.//varray[@name="forces"]')
            if forces_block is not None:
                forces = []
                for v in forces_block.findall('v'):
                    forces.append([float(x) for x in v.text.split()])
                data['forces'] = np.array(forces)
            
            # Get stress (optional)
            stress_block = calculation.find('.//varray[@name="stress"]')
            if stress_block is not None:
                stress = []
                for v in stress_block.findall('v'):
                    stress.append([float(x) for x in v.text.split()])
                data['stress'] = np.array(stress)
            
            if 'energy' in data and 'forces' in data:
                results.append(data)
        
        return results if results else None
        
    except Exception as e:
        logger.debug(f"Failed to read vasprun.xml {vasprun_path}: {e}")
        return None


def read_oszicar(oszicar_path: Path) -> Optional[List[float]]:
    """
    Read energies from OSZICAR file.
    
    Args:
        oszicar_path: Path to OSZICAR file
        
    Returns:
        List of energies (one per ionic step)
    """
    import re
    
    try:
        energies = []
        with open(oszicar_path, 'r') as f:
            for line in f:
                if ' F=' in line:
                    match = re.search(r'F=\s*([-.\dE+]+)', line)
                    if match:
                        energies.append(float(match.group(1)))
        
        return energies if energies else None
        
    except Exception as e:
        logger.debug(f"Failed to read OSZICAR {oszicar_path}: {e}")
        return None


def read_vasprun_forces_streaming(vasprun_path: Path, n_atoms: int) -> Optional[List[np.ndarray]]:
    """
    Read forces from vasprun.xml using streaming (line-by-line) parsing.
    
    This method works even if the XML file is incomplete (truncated).
    
    Args:
        vasprun_path: Path to vasprun.xml file
        n_atoms: Number of atoms per frame (to group force vectors)
        
    Returns:
        List of force arrays, one per ionic step
    """
    import re
    
    try:
        forces_all = []
        current_forces = []
        in_forces_block = False
        
        # Pattern to match force vectors
        v_pattern = re.compile(r'<v>\s*([-.\dE+]+)\s+([-.\dE+]+)\s+([-.\dE+]+)\s*</v>')
        
        with open(vasprun_path, 'r') as f:
            for line in f:
                if 'name="forces"' in line:
                    in_forces_block = True
                    current_forces = []
                elif in_forces_block:
                    if '</varray>' in line:
                        if len(current_forces) == n_atoms:
                            forces_all.append(np.array(current_forces))
                        in_forces_block = False
                        current_forces = []
                    else:
                        match = v_pattern.search(line)
                        if match:
                            force = [float(match.group(1)), float(match.group(2)), float(match.group(3))]
                            current_forces.append(force)
        
        return forces_all if forces_all else None
        
    except Exception as e:
        logger.debug(f"Failed to stream-parse vasprun.xml {vasprun_path}: {e}")
        return None


def merge_xdatcar_vasprun(
    structures: List[Atoms],
    properties: List[Dict[str, Any]]
) -> List[Atoms]:
    """
    Merge structures from XDATCAR with properties from vasprun.xml.
    
    Args:
        structures: List of Atoms from XDATCAR
        properties: List of property dicts from vasprun.xml
        
    Returns:
        List of Atoms with energy and forces attached
    """
    # Handle length mismatch
    n_struct = len(structures)
    n_props = len(properties)
    
    if n_struct != n_props:
        logger.warning(
            f"Structure count ({n_struct}) != property count ({n_props}). "
            f"Using minimum of both."
        )
    
    n_frames = min(n_struct, n_props)
    result = []
    
    for i in range(n_frames):
        atoms = structures[i].copy()
        props = properties[i]
        
        # Attach energy
        if 'energy' in props:
            atoms.info['energy'] = props['energy']
        
        # Attach forces
        if 'forces' in props:
            atoms.arrays['forces'] = props['forces']
        
        # Attach stress (optional)
        if 'stress' in props:
            atoms.info['stress'] = props['stress']
        
        result.append(atoms)
    
    return result


def convert_vasp_directory(vasp_dir: Path) -> Optional[List[Atoms]]:
    """
    Convert a single VASP directory to Atoms objects.
    
    Strategy:
    1. Try OUTCAR first (most reliable)
    2. If OUTCAR fails, use XDATCAR + vasprun.xml (full parse)
    3. If vasprun.xml is incomplete, use XDATCAR + OSZICAR + vasprun.xml (streaming)
    
    Args:
        vasp_dir: Path to VASP calculation directory
        
    Returns:
        List of Atoms objects, or None if conversion failed
    """
    outcar_path = vasp_dir / 'OUTCAR'
    xdatcar_path = vasp_dir / 'XDATCAR'
    vasprun_path = vasp_dir / 'vasprun.xml'
    oszicar_path = vasp_dir / 'OSZICAR'
    
    # Strategy 1: Try OUTCAR
    if outcar_path.exists():
        atoms_list = read_outcar(outcar_path)
        if atoms_list is not None:
            logger.debug(f"Successfully read {len(atoms_list)} frames from OUTCAR: {vasp_dir}")
            return atoms_list
        else:
            logger.info(f"OUTCAR exists but failed to read, trying fallback: {vasp_dir}")
    
    # Strategy 2: XDATCAR + vasprun.xml fallback (full XML parse)
    if xdatcar_path.exists() and vasprun_path.exists():
        structures = read_xdatcar(xdatcar_path)
        properties = read_vasprun_xml(vasprun_path)
        
        if structures is not None and properties is not None:
            atoms_list = merge_xdatcar_vasprun(structures, properties)
            if atoms_list:
                logger.debug(f"Successfully merged {len(atoms_list)} frames from XDATCAR+vasprun.xml: {vasp_dir}")
                return atoms_list
        
        # Strategy 3: XDATCAR + OSZICAR + vasprun.xml streaming (for incomplete XML)
        if structures is not None and oszicar_path.exists():
            logger.info(f"vasprun.xml parse failed, trying streaming fallback: {vasp_dir}")
            
            energies = read_oszicar(oszicar_path)
            n_atoms = len(structures[0]) if structures else 0
            forces = read_vasprun_forces_streaming(vasprun_path, n_atoms) if n_atoms > 0 else None
            
            if energies is not None and forces is not None:
                # Create property dicts
                n_frames = min(len(structures), len(energies), len(forces))
                logger.info(f"Streaming fallback: {len(structures)} structures, {len(energies)} energies, {len(forces)} force frames -> {n_frames} complete frames")
                
                properties = []
                for i in range(n_frames):
                    properties.append({
                        'energy': energies[i],
                        'forces': forces[i],
                    })
                
                atoms_list = merge_xdatcar_vasprun(structures[:n_frames], properties)
                if atoms_list:
                    logger.debug(f"Successfully merged {len(atoms_list)} frames using streaming fallback: {vasp_dir}")
                    return atoms_list
    
    # No valid data found
    logger.debug(f"No valid VASP data found in: {vasp_dir}")
    return None


def convert_vasp_to_extxyz(
    root_dir: str,
    output_path: str,
    recursive: bool = True,
    include_stress: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Convert all VASP calculations in a directory tree to a single extxyz file.
    
    Args:
        root_dir: Root directory to search for VASP calculations
        output_path: Output path for extxyz file
        recursive: If True, search subdirectories recursively
        include_stress: If True, include stress tensor in output
        verbose: If True, show progress bar
        
    Returns:
        Dictionary with conversion statistics
    """
    root = Path(root_dir)
    output = Path(output_path)
    
    # Find all VASP directories
    logger.info(f"Searching for VASP calculations in: {root}")
    vasp_dirs = find_vasp_directories(root_dir)
    logger.info(f"Found {len(vasp_dirs)} potential VASP directories")
    
    all_atoms = []
    stats = {
        'total_directories': len(vasp_dirs),
        'successful': 0,
        'failed': 0,
        'total_frames': 0,
        'failed_dirs': [],
    }
    
    # Process each directory
    iterator = tqdm(vasp_dirs, desc="Converting VASP") if verbose else vasp_dirs
    
    for vasp_dir in iterator:
        try:
            atoms_list = convert_vasp_directory(vasp_dir)
            
            if atoms_list is not None and len(atoms_list) > 0:
                all_atoms.extend(atoms_list)
                stats['successful'] += 1
                stats['total_frames'] += len(atoms_list)
            else:
                stats['failed'] += 1
                stats['failed_dirs'].append(str(vasp_dir))
                
        except Exception as e:
            logger.warning(f"Error processing {vasp_dir}: {e}")
            stats['failed'] += 1
            stats['failed_dirs'].append(str(vasp_dir))
    
    # Write output
    if all_atoms:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_extxyz(all_atoms, output, include_stress=include_stress)
        logger.info(f"Wrote {len(all_atoms)} frames to {output}")
    else:
        logger.warning("No valid data found to write")
    
    return stats


def write_extxyz(
    atoms_list: List[Atoms],
    output_path: Path,
    include_stress: bool = False,
) -> None:
    """
    Write atoms to extxyz format with proper formatting.
    
    Args:
        atoms_list: List of Atoms objects
        output_path: Output file path
        include_stress: Include stress tensor
    """
    with open(output_path, 'w') as f:
        for atoms in atoms_list:
            n_atoms = len(atoms)
            
            # Build comment line
            cell_str = " ".join(f"{x:.8f}" for x in atoms.cell.flatten())
            
            # Properties string
            has_forces = 'forces' in atoms.arrays
            if has_forces:
                props = "species:S:1:pos:R:3:forces:R:3"
            else:
                props = "species:S:1:pos:R:3"
            
            # Energy
            energy = atoms.info.get('energy', 0.0)
            
            # Comment line
            comment = f'Lattice="{cell_str}" Properties={props} energy={energy:.8f} pbc="T T T"'
            
            # Add stress if present and requested
            if include_stress and 'stress' in atoms.info:
                stress = atoms.info['stress']
                if hasattr(stress, 'flatten'):
                    stress_str = " ".join(f"{x:.8f}" for x in stress.flatten())
                    comment += f' stress="{stress_str}"'
            
            f.write(f"{n_atoms}\n")
            f.write(f"{comment}\n")
            
            # Write atoms
            symbols = atoms.get_chemical_symbols()
            positions = atoms.positions
            forces = atoms.arrays.get('forces', np.zeros((n_atoms, 3)))
            
            for i in range(n_atoms):
                if has_forces:
                    f.write(f"{symbols[i]:4s} {positions[i,0]:16.8f} {positions[i,1]:16.8f} {positions[i,2]:16.8f} {forces[i,0]:16.8f} {forces[i,1]:16.8f} {forces[i,2]:16.8f}\n")
                else:
                    f.write(f"{symbols[i]:4s} {positions[i,0]:16.8f} {positions[i,1]:16.8f} {positions[i,2]:16.8f}\n")


# CLI interface
def main():
    """Command-line interface for VASP to extxyz conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert VASP calculations to extxyz format'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Root directory containing VASP calculations'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='dataset.xyz',
        help='Output extxyz file path (default: dataset.xyz)'
    )
    parser.add_argument(
        '--include-stress',
        action='store_true',
        help='Include stress tensor in output'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run conversion
    stats = convert_vasp_to_extxyz(
        root_dir=args.input_dir,
        output_path=args.output,
        include_stress=args.include_stress,
        verbose=True,
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print("Conversion Summary")
    print("=" * 50)
    print(f"Total directories scanned: {stats['total_directories']}")
    print(f"Successfully converted: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total frames: {stats['total_frames']}")
    
    if stats['failed_dirs'] and args.verbose:
        print("\nFailed directories:")
        for d in stats['failed_dirs'][:10]:
            print(f"  - {d}")
        if len(stats['failed_dirs']) > 10:
            print(f"  ... and {len(stats['failed_dirs']) - 10} more")


if __name__ == '__main__':
    main()
