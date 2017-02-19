# -*- sieve -*-
# This file is part of Mailutils testsuite.
# Copyright (C) 2003, 2010 Free Software Foundation, Inc.
# See file COPYING for distribution conditions.

require ["relational", "comparator-i;ascii-numeric", "fileinto"];

if header :value "lt" :comparator "i;ascii-numeric" ["x-priority"] ["3"]
  {
    fileinto "%Priority";
  }
elsif address :count "gt" :comparator "i;ascii-numeric" ["to"] ["5"]
  {
    # everything with more than 5 recipients in the "to" field
    # is considered SPAM
    fileinto "%SPAM";
  }
elsif address :value "gt" :all :comparator "i;ascii-casemap" ["from"] ["M"]
  {
    fileinto "%From_N-Z";
  }
else
  {
    fileinto "%From_A-M";
  }

if allof (address :count "eq" :comparator "i;ascii-numeric"
                  ["to", "cc"] ["1"] ,
          address :all :comparator "i;ascii-casemap"
                  ["to", "cc"] ["me@foo.example.com.invalid"])
  {
    fileinto "%Only_me";
  }

# End of rel-hairy.sv
