;;;; -*- scheme -*-
;;;; GNU Mailutils -- a suite of utilities for electronic mail
;;;; Copyright (C) 2002, 2003, 2004, 2006, 2007, 2009, 2010 Free
;;;; Software Foundation, Inc.
;;;;
;;;; GNU Mailutils is free software; you can redistribute it and/or modify
;;;; it under the terms of the GNU General Public License as published by
;;;; the Free Software Foundation; either version 3, or (at your option)
;;;; any later version.
;;;; 
;;;; GNU Mailutils is distributed in the hope that it will be useful,
;;;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;;;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;;;; GNU General Public License for more details.
;;;; 
;;;; You should have received a copy of the GNU General Public License along
;;;; with GNU Mailutils; if not, write to the Free Software Foundation,
;;;; Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
;;;;

(define-module (mailutils mailutils)
  #:use-module (ice-9 documentation))

(set! documentation-files (append documentation-files 
                                  (list "/usr/local/share/guile/site/mailutils/guile-procedures.txt")))

(define mu-libs (list "libmailutils"
		      "libmu_auth"
		      "libmu_mbox"
		      "libmu_mh"
		      "libmu_maildir"
		      "libmu_pop"
		      "libmu_imap"))

(let ((lib-path "/usr/local/lib/"))
  (for-each
   (lambda (lib)
	   (dynamic-link (string-append lib-path lib)))
   mu-libs)
  (load-extension (string-append
		   lib-path "libguile-mailutils-v-2.1.90") "mu_scm_init"))

;;;; End of mailutils.scm
