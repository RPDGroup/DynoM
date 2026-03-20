
import io
import gzip
import copy
import functools
import collections
import dataclasses
from pathlib import Path, PosixPath
from typing import Any, Mapping, Sequence, Union
import pprint
from Bio import PDB

import residue_constants


MmCIFDict = Mapping[str, Sequence[str]]


@dataclasses.dataclass(frozen=True)
class Monomer:
    id: str
    num: int


@dataclasses.dataclass(frozen=True)
class AtomSite:
    id: str # id, unique identifier
    type_symbol: str # type_symbol
    model_num: str # pdbx_PDB_model_num

    author_chain_id: str # auth_asym_id
    author_atom_id: str # auth_atom_id
    author_residue_name: str # auth_comp_id
    author_seq_num: str # auth_seq_id

    mmcif_chain_id: str # label_asym_id
    mmcif_atom_id: str # label_atom_id
    mmcif_residue_name: str # label_comp_id
    mmcif_seq_num: str # label_seq_id
    mmcif_entity_id: str # label_entity_id

    hetatm_atom: str # group_PDB
    mmcif_alt_id: str # label_alt_id
    insertion_code: str # pdbx_PDB_ins_code
    
    b_factor: str # B_iso_or_equiv
    occupancy: str # occupancy
    coord_x: str # Cartn_x
    coord_y: str # Cartn_y
    coord_z: str # Cartn_z


# Used to map SEQRES index to a residue in the structure.
@dataclasses.dataclass(frozen=True)
class ResidueAtPosition:
    name: str
    resseq: int
    res_alt:str
    is_missing: bool


@dataclasses.dataclass(frozen=True)
class MmcifObject:
    pdb_id: str
    header: Mapping[str, Any]
    full_structure: PDB.Structure.Structure # Biopython structure
    chain_to_seqres: Mapping[str, str] # chain_id -> seqres
    entity_to_chains: Mapping[str, Sequence[str]] # entity_id -> list of chain IDs
    struct_mappings: Mapping[int, Mapping[str, Mapping[int, ResidueAtPosition]]] # model_id -> chain_id -> seq_idx -> ResidueAtPosition


def mmcif_loop_to_list(prefix: str,
                       parsed_info: MmCIFDict) -> Sequence[Mapping[str, str]]:
    cols = []
    data = []
    for key, value in parsed_info.items():
        if key.startswith(prefix):
            cols.append(key)
            data.append(value)

    assert all([len(xs) == len(data[0]) for xs in data]), (
        'mmCIF error: Not all loops are the same length: %s' % cols)

    return [dict(zip(cols, xs)) for xs in zip(*data)]


def mmcif_loop_to_dict(prefix: str,
                       index: str,
                       parsed_info: MmCIFDict,
                       ) -> Mapping[str, Mapping[str, str]]:
    """Extracts loop associated with a prefix from mmCIF data as a dictionary."""
    entries = mmcif_loop_to_list(prefix, parsed_info)
    return {entry[index]: entry for entry in entries}


def recursive_defaultdict():
    return collections.defaultdict(recursive_defaultdict)

@functools.lru_cache(16, typed=False)
def parse(
    mmcif_path: Union[str, PosixPath],
) -> MmcifObject:
    """Entry point, parses an mmcif_string.

    Args:
        mmcif_path: Path to mmcif file.

    Returns:
        MmcifObject
    """
    if isinstance(mmcif_path, str):
        mmcif_path = Path(mmcif_path)
    assert mmcif_path.is_file(), f'Cannot find mmCIF file at {mmcif_path}.'
    pdb_id, _, _ = mmcif_path.name.split('.')
    assert len(pdb_id) == 4
    with gzip.open(mmcif_path, 'rt') as f:
        mmcif_string = f.read()
    parser = PDB.MMCIFParser(QUIET=True)
    handle = io.StringIO(mmcif_string)
    full_structure = parser.get_structure('', handle)
    parsed_info = parser._mmcif_dict 
    for key, value in parsed_info.items():
        if not isinstance(value, list):
            parsed_info[key] = [value]

    header = _get_header(parsed_info)

    # Determine the protein chains, and their start numbers according to the
    # internal mmCIF numbering scheme (likely but not guaranteed to be 1).
    valid_mmcif_chains, entity_to_mmcif_chains = _get_protein_chains(parsed_info=parsed_info)
    if len(valid_mmcif_chains) == 0:
        raise Exception('No protein chains found in this file.')
    seq_start_num = {mmcif_chain_id: min(polymer.keys())
                        for mmcif_chain_id, polymer in valid_mmcif_chains.items()}

    mmcif_to_author_chain_id = {}
    polymer_to_structure_mappings = recursive_defaultdict()
    atomsite_dict = _get_atomsite_dict(parsed_info)

    has_other_atom=False
    for _, atomsite in atomsite_dict.items():

        # -mmcif_to_author_chain_id
        if atomsite.mmcif_chain_id in valid_mmcif_chains:
            # assure chain id mapping consistency 
            if mmcif_to_author_chain_id.get(atomsite.mmcif_chain_id, None) is not None:
                assert  mmcif_to_author_chain_id[atomsite.mmcif_chain_id] == atomsite.author_chain_id, \
                    'Author chain id does not match mmcif chain id.'
            else:
                mmcif_to_author_chain_id[atomsite.mmcif_chain_id] = atomsite.author_chain_id

           ## 
            if atomsite.type_symbol != "CA" :
                has_other_atom=True

            if atomsite.hetatm_atom == 'HETATM': continue
            assert atomsite.hetatm_atom == 'ATOM',  \
                "hetatm_atom != 'ATOM'"
            # skip altloc atoms
            if _is_set(atomsite.mmcif_alt_id): continue
            # skip atoms with non-empty insertion code
            # if _is_set(atomsite.insertion_code): continue
            # skip atoms with low occupancy
            if float(atomsite.occupancy) < 0.5: continue


            # assure residue name consistency
            assert atomsite.mmcif_residue_name == atomsite.author_residue_name, \
                'MmCIF/author residue name mistach.'
            # skip non-standard amino acids
            if atomsite.mmcif_residue_name not in residue_constants.restype_3to1: continue
            # assure seqres mapping consistency 
            if valid_mmcif_chains[atomsite.mmcif_chain_id][int(atomsite.mmcif_seq_num)].id != \
                atomsite.mmcif_residue_name: continue

            seq_idx = int(atomsite.mmcif_seq_num) - seq_start_num[atomsite.mmcif_chain_id] # 0-based indexing for seqres mapping
            current_model = polymer_to_structure_mappings.get(atomsite.model_num, {})
            current_chain = current_model.get(atomsite.author_chain_id, {})
            if current_chain.get(seq_idx, None) is not None:
                assert current_chain[seq_idx].name == atomsite.mmcif_residue_name, 'Residue name mismatch.'
                assert current_chain[seq_idx].resseq == int(atomsite.author_seq_num), 'Residue number mismatch.'
                continue
            
            current_chain[seq_idx] = ResidueAtPosition(name=atomsite.mmcif_residue_name,
                                                       resseq=int(atomsite.author_seq_num),
                                                       res_alt=atomsite.insertion_code,
                                                       is_missing=False)

            polymer_to_structure_mappings[atomsite.model_num][atomsite.author_chain_id] = current_chain
    if len(polymer_to_structure_mappings) == 0:
        raise Exception('No valid chains found in this file.')
    if not has_other_atom :
        raise Exception('all atom are CA not other.')

    for _, current_model in polymer_to_structure_mappings.items():
        for mmcif_chain_id, polymer in valid_mmcif_chains.items():
            author_chain_id = mmcif_to_author_chain_id[mmcif_chain_id]
            current_chain = current_model.get(author_chain_id, None)
            if current_chain is None: continue
            for mmcif_seq_num, monomer in polymer.items():
                seq_idx = mmcif_seq_num - seq_start_num[mmcif_chain_id]
                if seq_idx not in current_chain:
                    current_chain[seq_idx] = ResidueAtPosition(name='UNK',
                                                               resseq=-1, # invalid resseq
                                                               res_alt=-1,
                                                               is_missing=True)

            assert len(current_chain) == len(polymer), 'len(current_chain) != len(polymer)'
            for seq_idx, curr_resi in current_chain.items():
                if curr_resi.is_missing: continue
                mmcif_seq_num = seq_idx + seq_start_num[mmcif_chain_id]
                assert curr_resi.name == polymer[mmcif_seq_num].id, 'curr_resi.name != polymer[mmcif_seq_num].id'
                assert curr_resi.name in residue_constants.restype_3to1, 'Invalid residue name.'

    author_chain_to_seqres = {}
    for mmcif_chain_id, polymer in valid_mmcif_chains.items():
        author_chain_id = mmcif_to_author_chain_id[mmcif_chain_id]
        seqres = []
        for _, monomer in polymer.items():
            seqres.append(residue_constants.restype_3to1.get(monomer.id, 'X'))
        seqres = ''.join(seqres)
        author_chain_to_seqres[author_chain_id] = seqres

    # entity to author chains
    entity_to_author_chains = collections.defaultdict(set)
    for entity_id, mmcif_chain_list in entity_to_mmcif_chains.items():
        for mmcif_chain in mmcif_chain_list:
            entity_to_author_chains[entity_id].add(mmcif_to_author_chain_id[mmcif_chain])
    entity_to_author_chains = {k: list(v) for k, v in entity_to_author_chains.items()}

    # check model consistency between Biopython and mmCIF
    mmcif_model_ids = [int(model_id) for model_id in polymer_to_structure_mappings] # mmCIF model ids are strings and not necessarily start from 1
    bio_model_ids = [model_id for model_id in full_structure.child_dict if isinstance(model_id, int)] # biopython model ids are zero-based integers
    assert len(mmcif_model_ids) == len(bio_model_ids), 'Inconsistent number of models.'
    assert mmcif_model_ids == sorted(mmcif_model_ids), 'Unsorted mmcif model ids.'
    assert bio_model_ids == sorted(bio_model_ids), 'Unsorted biopython structure model ids.'
    assert 0 in full_structure.child_dict.keys(), '0 not in biopython model keys.'
    
    # update polymer_to_structure_mappings key from mmcif_model_id to bio_model_id
    struct_mappings = {}
    for mmcif_model_id, mapping in polymer_to_structure_mappings.items():
        # print("mmcif_model_id, mapping :",mmcif_model_id, mapping )
        bio_model_id = int(mmcif_model_id) - min(mmcif_model_ids) # mmcif_model_id not necessarily start from 1
        assert bio_model_id in full_structure.child_dict, 'Model id mismatch.'
        struct_mappings[bio_model_id] = copy.deepcopy(mapping)

    mmcif_object = MmcifObject(
        pdb_id=pdb_id.upper(),
        header=header,
        full_structure=full_structure, # Biopython structure
        chain_to_seqres=author_chain_to_seqres, # chain_id -> seqres
        entity_to_chains=entity_to_author_chains, # entity_id -> list of chain IDs
        struct_mappings=struct_mappings, # model_id -> chain_id -> seq_idx -> ResidueAtPosition
        
    )
    reverse_mapping = {v: k for k, v in mmcif_to_author_chain_id.items()}
    return mmcif_object,reverse_mapping #author_chain_id_to_mmcif->metadata_use

_MIN_LENGTH_OF_CHAIN_TO_BE_COUNTED_AS_PEPTIDE = 1


def get_release_date(parsed_info: MmCIFDict) -> str:
    """Returns the oldest revision date."""
    revision_dates = parsed_info['_pdbx_audit_revision_history.revision_date']
    return min(revision_dates)


def _get_header(parsed_info: MmCIFDict) -> Mapping[str, Any]:
    """Returns a basic header containing method, release date and resolution."""
    header = {}

    experiments = mmcif_loop_to_list('_exptl.', parsed_info)
    header['structure_method'] = ','.join([
        experiment['_exptl.method'].lower() for experiment in experiments])

    if '_pdbx_audit_revision_history.revision_date' in parsed_info:
        header['release_date'] = get_release_date(parsed_info)
    else:
        raise Exception('Could not determine release_date.')

    header['resolution'] = 0.00
    for res_key in ('_refine.ls_d_res_high', '_em_3d_reconstruction.resolution',
                    '_reflns.d_resolution_high'):
        if res_key in parsed_info:
            try:
                raw_resolution = float(parsed_info[res_key][0])
                header['resolution'] = raw_resolution
            except:
                pass

    return header


def _get_atomsite_dict(parsed_info: MmCIFDict) -> Mapping[str, AtomSite]:
    """Returns a dictionary with atom site info; contains data not present in the structure."""
    
    atomsite_list = [
        AtomSite(*site_info) for site_info in zip(  # pylint:disable=g-complex-comprehension
            parsed_info['_atom_site.id'],
            parsed_info['_atom_site.type_symbol'],
            parsed_info['_atom_site.pdbx_PDB_model_num'],

            parsed_info['_atom_site.auth_asym_id'],
            parsed_info['_atom_site.auth_atom_id'],
            parsed_info['_atom_site.auth_comp_id'],
            parsed_info['_atom_site.auth_seq_id'],

            parsed_info['_atom_site.label_asym_id'],
            parsed_info['_atom_site.label_atom_id'],
            parsed_info['_atom_site.label_comp_id'],
            parsed_info['_atom_site.label_seq_id'],
            parsed_info['_atom_site.label_entity_id'],
            
            parsed_info['_atom_site.group_PDB'],
            parsed_info['_atom_site.label_alt_id'],
            parsed_info['_atom_site.pdbx_PDB_ins_code'],

            parsed_info['_atom_site.B_iso_or_equiv'],
            parsed_info['_atom_site.occupancy'],
            parsed_info['_atom_site.Cartn_x'],
            parsed_info['_atom_site.Cartn_y'],
            parsed_info['_atom_site.Cartn_z'],
        )
    ]
    
    return {site_info.id: site_info for site_info in atomsite_list}


def _get_protein_chains(
        *,
        parsed_info: Mapping[str, Any]
):

    entity_poly_seqs = mmcif_loop_to_list('_entity_poly_seq.', parsed_info)

    polymers = collections.defaultdict(dict)
    for entity_poly_seq in entity_poly_seqs:
        ptr = polymers[entity_poly_seq['_entity_poly_seq.entity_id']]
        ptr[int(entity_poly_seq['_entity_poly_seq.num'])] = \
            Monomer(
                id=entity_poly_seq['_entity_poly_seq.mon_id'],
                num=int(entity_poly_seq['_entity_poly_seq.num'])
            )
    for _, polymer in polymers.items():
        assert len(polymer) == max(polymer.keys()) - min(polymer.keys()) + 1, \
            'Inconsistent polymer seqres length.'

    chem_comps = mmcif_loop_to_dict(
        '_chem_comp.', '_chem_comp.id', parsed_info)

    struct_asyms = mmcif_loop_to_list('_struct_asym.', parsed_info)

    entity_to_mmcif_chains = collections.defaultdict(set)
    for struct_asym in struct_asyms:
        mmcif_chain_id = struct_asym['_struct_asym.id']
        entity_id = struct_asym['_struct_asym.entity_id']
        entity_to_mmcif_chains[entity_id].add(mmcif_chain_id)

    # Identify and return valid protein chains.
    valid_mmcif_chains = {}
    for entity_id, seq_info in polymers.items():
        mmcif_chain_ids = entity_to_mmcif_chains[entity_id]

        # Reject polymers without any peptide-like components, such as DNA/RNA.
        if sum(['peptide' in chem_comps[monomer.id]['_chem_comp.type']
                for _, monomer in seq_info.items()]) >= _MIN_LENGTH_OF_CHAIN_TO_BE_COUNTED_AS_PEPTIDE:
            for mmcif_chain_id in mmcif_chain_ids:
                valid_mmcif_chains[mmcif_chain_id] = seq_info

    # only keep valid chains for entity-chain mapping
    for entity_id, mmcif_chain_list in entity_to_mmcif_chains.items():
        entity_to_mmcif_chains[entity_id] = [chain for chain in mmcif_chain_list if chain in valid_mmcif_chains]
        assert len(entity_to_mmcif_chains[entity_id]) == len(set(entity_to_mmcif_chains[entity_id])), \
            'Duplicate chain in entity-chain mapping.'

    return valid_mmcif_chains, entity_to_mmcif_chains


def _is_set(data: str) -> bool:
    """Returns False if data is a special mmCIF character indicating 'unset'."""
    return data not in ('.', '?')
