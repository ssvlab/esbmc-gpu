# -*- sieve -*-
# This file is part of Mailutils testsuite.
# Copyright (C) 2002, 2010 Free Software Foundation, Inc.
# See file COPYING for distribution conditions.

require "comparator-i;ascii-casemap";

if header :comparator "i;ascii-casemap" :matches "subject" "*you, too,*"
  {
    discard;
  }
