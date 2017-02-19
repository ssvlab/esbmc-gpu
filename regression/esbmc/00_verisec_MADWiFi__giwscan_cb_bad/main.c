#include "../constants.h"

static int
giwscan_cb(const struct ieee80211_scan_entry *se)
{
  char buf[IESZ];

  /* Everything up to this point seems irrelevant to the following. */

  if (se->se_rsn_ie != NULL) {
    /* BAD */
    assert(se->se_rsn_ie[1] + 2 <= IESZ);
    r_memcpy(buf, se->se_rsn_ie, se->se_rsn_ie[1] + 2);
  }
  
  return 0;
}

int main ()
{
  struct ieee80211_scan_entry se;
  u_int8_t ie [IESZ * 2];
  se.se_rsn_ie = ie;
  se.se_rsn_ie[1] = (IESZ * 2) + 2;

  giwscan_cb (&se);

  return 0;
}
