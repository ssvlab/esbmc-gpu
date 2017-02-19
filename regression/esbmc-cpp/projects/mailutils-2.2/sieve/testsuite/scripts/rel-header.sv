# -*- sieve -*-
# This file is part of Mailutils testsuite.
# Copyright (C) 2003, 2010 Free Software Foundation, Inc.
# See file COPYING for distribution conditions.

require ["relational", "comparator-i;ascii-numeric"];

if header :count "gt" ["received"] ["2"]
  {
    discard;
  }

# End of rel-address.sv
