#include "c_autodiff.h"

DualNumber dualnumber_initialise(int ndiff, double value, double diff) {
    DualNumber dn;
    dn.ndiff = ndiff <= AUTODIFF_NDIFFMAX ? ndiff : AUTODIFF_NDIFFMAX;
    dn.value = value;
    for(int i = 0; i < ndiff; i ++)
        dn.diff[i] = diff;
    return dn;
}

int get_ndiff(DualNumber dn1, DualNumber dn2) {
    return dn1.ndiff <= dn2.ndiff ? dn1.ndiff : dn2.ndiff;
}

DualNumber dualnumber_add(DualNumber dn1, DualNumber dn2) {
    DualNumber dn;
    dn.ndiff = get_ndiff(dn1, dn2);
    dn.value = dn1.value + dn2.value;
    for(int i = 0; i < dn.ndiff; i ++)
        dn.diff[i] = dn1.diff[i] + dn2.diff[i];
    return dn;
}

DualNumber dualnumber_diff(DualNumber dn1, DualNumber dn2) {
    DualNumber dn;
    dn.ndiff = get_ndiff(dn1, dn2);
    dn.value = dn1.value - dn2.value;
    for(int i = 0; i < dn.ndiff; i ++)
        dn.diff[i] = dn1.diff[i] - dn2.diff[i];
    return dn;
}

DualNumber dualnumber_mult(DualNumber dn1, DualNumber dn2) {
    DualNumber dn;
    dn.ndiff = get_ndiff(dn1, dn2);
    dn.value = dn1.value * dn2.value;
    for(int i = 0; i < dn.ndiff; i ++)
        dn.diff[i] = dn1.diff[i] * dn2.value
            + dn2.diff[i] * dn1.value;
    return dn;
}

DualNumber dualnumber_div(DualNumber dn1, DualNumber dn2) {
    DualNUmber dn;
    dn.ndiff = get_ndiff(dn1, dn2);
    dn.value = dn1.value / dn2.value;
    for(int i = 0; i < dn.ndiff; i ++)
        dn.diff[i] = (dn1.diff[i] - dn.value * dn2.diff[i])
            / dn2.value;
    return dn;
}

DualNumber dualnumber_exp(DualNumber dn1) {
    DualNUmber dn;
    dn.ndiff = dn1.ndiff;
    dn.value = exp(dn1.value);
    for(int i = 0; i < dn.ndiff; i ++)
        dn.diff[i] =  dn1.diff[i] * dn.value;
    return dn;
}

DualNumber dualnumber_log(DualNumber dn1) {
    DualNUmber dn;
    dn.ndiff = dn1.ndiff;
    dn.value = log(dn1.value);
    for(int i = 0; i < dn.ndiff; i ++)
        dn.diff[i] =  dn1.diff[i] / dn1.value;
    return dn;
}

DualNumber dualnumber_sqrt(DualNumber dn1) {
    DualNUmber dn;
    dn.ndiff = dn1.ndiff;
    dn.value = sqrt(dn1.value);
    for(int i = 0; i < dn.ndiff; i ++)
        dn.diff[i] =  dn1.diff[i] / 2. / dn.value;
    return dn;
}


