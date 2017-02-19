/*
  fn1 returns a positive value less than n.
*/
int fn1(int n) {
    int ret = nondet_int(); //__NONDET__();
    __ESBMC_assume(0 <= ret);
    __ESBMC_assume(ret < n);
    return ret;
}

int main() {
    int i1 = fn1(2); /* 0 or 1 */
    int i2 = fn1(2); /* 0 or 1 */
    int i3 = fn1(2); /* 0 or 1 */
    int i4 = fn1(2); /* 0 or 1 */
    int i5 = fn1(2); /* 0 or 1 */
    int i6 = fn1(2); /* 0 or 1 */
    int i7 = fn1(2); /* 0 or 1 */
    int i8 = fn1(2); /* 0 or 1 */
    assert(i1 + i2 + i3 + i4 + i5 + i6 + i7 + i8 <= 8); /* should be proved */
    return 0;
}
