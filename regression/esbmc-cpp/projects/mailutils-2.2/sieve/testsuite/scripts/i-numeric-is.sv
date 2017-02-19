# -*- sieve -*-
# This file is part of Mailutils testsuite.
# Copyright (C) 2002, 2010 Free Software Foundation, Inc.
# See file COPYING for distribution conditions.

require "comparator-i;ascii-numeric";

if header :comparator "i;ascii-numeric" :is "X-Number" "15"
  {
    discard;
  }
