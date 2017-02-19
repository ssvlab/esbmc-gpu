;;;; GNU Mailutils -- a suite of utilities for electronic mail
;;;; Copyright (C) 1999, 2000, 2001, 2007, 2010 Free Software
;;;; Foundation, Inc.
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

;;;; This module provides GNU extension test "numaddr".

;;;; Syntax:   numaddr [":over" / ":under"] <header-names: string-list>
;;;;           <limit: number>
;;;; The "numaddr" test counts Internet addresses in structured headers
;;;; that contain addresses.  It returns true if the total number of
;;;; addresses satisfies the requested relation:
;;;;
;;;; If the argument is ":over" and the number of addresses is greater than
;;;; the number provided, the test is true; otherwise, it is false.
;;;;
;;;; If the argument is ":under" and the number of addresses is less than
;;;; the number provided, the test is true; otherwise, it is false.
;;;;
;;;; If the argument is empty, ":over" is assumed.

;;;; Example:
;;;;
;;;; require [ "numaddr" ];
;;;; if numaddr :over [ "To", "Cc" ] 50 { discard; }

(define (test-numaddr header-list count . comp)
  (let ((total-count 0)
	(header-fields (mu-message-get-header-fields
			sieve-current-message
			header-list))
	(compfun (cond
		  ((or (null? (car comp)) (eq? (car comp) #:over))
		   (lambda (val lim)
		     (> val lim)))
		  ((eq? (car comp) #:under)
		   (lambda (val lim)
		     (< val lim)))
		  (else
		   (runtime-message SIEVE-ERROR
                                  "test-numaddr: unknown comparator "
				  comp)))))
    (call-with-current-continuation
     (lambda (exit)
       (for-each
	(lambda (h)
	  (let ((hdr (cdr h)))
	    (if hdr
		(let ((naddr (mu-address-get-count hdr)))
		  (set! total-count (+ total-count naddr))
		  (if (compfun total-count count)
		      (exit #t))))))
	header-fields)
       #f))))

;;; Register the test at compile time
(if sieve-parser
    (sieve-register-test "numaddr"
			 test-numaddr
			 (list 'string-list 'number)
			 size-comp))
