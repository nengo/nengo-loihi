import numpy as np

from .loihi_api import Board, Chip, Core, CxProfile, VthProfile


def compute_profiles(core, list_profiles):
    profile_lists = []
    for group in core.groups:
        profile_lists.append(list_profiles(group))

    profiles = list(set(p for plist in profile_lists for p in plist))
    profile_idxs = {}
    for group, plist in zip(core.groups, profile_lists):
        profile_idxs[group] = np.zeros(len(plist), dtype=int)
        for k, profile in enumerate(profiles):
            profile_idxs[group][[p == profile for p in plist]] = k

    return profiles, profile_idxs


def core_cx_profiles(core):
    """Compute all cxProfiles needed for a core"""
    def list_cx_profiles(group):
        profiles = []
        for i in range(group.n):
            profiles.append(CxProfile(
                decayU=group.decayU[i],
                decayV=group.decayV[i],
                refDelay=group.refDelay[i],
            ))

        return profiles

    return compute_profiles(core, list_cx_profiles)


def core_vth_profiles(core):
    """Compute all vthProfiles needed for a core"""
    def list_vth_profiles(group):
        profiles = []
        for i in range(group.n):
            profiles.append(VthProfile(
                vth=group.vth[i],
            ))

        return profiles

    return compute_profiles(core, list_vth_profiles)


# def single_synapse_fmt(synapses):
    # return


def one_to_one_allocator(cx_model):
    board = Board()
    chip = board.new_chip()

    for group in cx_model.groups:
        core = chip.new_core()
        core.add_group(group)

        cx_profiles, cx_profile_idxs = core_cx_profiles(core)
        [core.add_cx_profile(cx_profile) for cx_profile in cx_profiles]
        core.cxProfileIdxs = cx_profile_idxs

        vth_profiles, vth_profile_idxs = core_vth_profiles(core)
        [core.add_vth_profile(vth_profile) for vth_profile in vth_profiles]
        core.vthProfileIdxs = vth_profile_idxs

        for syn in group.synapses:
            core.add_synapses(chip, core, syn)


    return board
