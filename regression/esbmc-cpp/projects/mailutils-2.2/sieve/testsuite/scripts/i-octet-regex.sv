# -*- sieve -*-
# This file is part of Mailutils testsuite.
# Copyright (C) 2002, 2010 Free Software Foundation, Inc.
# See file COPYING for distribution conditions.

require "comparator-i;octet";

if header :comparator "i;octet" :regex "subject" ".*you.*"
  {
    discard;
  }
