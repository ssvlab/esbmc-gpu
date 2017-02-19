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

;;;; This module provides GNU extension test "mimeheader".

;;;; Syntax:   mimeheader [COMPARATOR] [MATCH-TYPE]
;;;;           <header-names: string-list> <key-list: string-list>
;;;;
;;;; The "mimeheader" test evaluates to true if in any part of the
;;;; multipart MIME message a header name from <header-names> list
;;;; matches any key from <key-list>. If the message is not multipart,
;;;; "mimeheader" test is equivalent to "header" test.
;;;;
;;;; The arguments to "mimeheader" test are the same as to "header" test.

;;;; Example:
;;;;
;;;; require [ "mimeheader", "reject"];
;;;; if mimeheader :matches "Content-Type" "*application/msword;*" {
;;;;     reject "Please do not send data in a proprietary format.";
;;;; }

(define (test-mimeheader header-list key-list . opt-args)
  (if (mu-message-multipart? sieve-current-message)
      (let ((mime-count (mu-message-get-num-parts sieve-current-message))
	    (comp (find-comp opt-args))
	    (match (find-match opt-args)))
	(call-with-current-continuation
	 (lambda (exit)
	   (do ((n 1 (1+ n)))
	       ((> n mime-count) #f)
	     (let ((msg (mu-message-get-part sieve-current-message n)))
	       (if msg
		   (for-each
		    (lambda (key)
		      (let ((header-fields (mu-message-get-header-fields
					    msg
					    header-list))
			    (rx (if (eq? match #:matches)
				    (make-regexp (sieve-regexp-to-posix key)
						 (if (eq? comp string-ci=?)
						     regexp/icase
						     '()))
				    #f)))
			(for-each
			 (lambda (h)
			   (let ((hdr (cdr h)))
			     (if hdr
				 (case match
				   ((#:is)
				    (if (comp hdr key)
					(exit #t)))
				   ((#:contains)
				    (if (sieve-str-str hdr key comp)
					(exit #t)))
				   ((#:matches)
				    (if (regexp-exec rx hdr)
					(exit #t)))))))
			 header-fields)))
		    key-list)
		   #f))))))
      (apply test-header header-list key-list opt-args)))

;;; Register the test at compile time
(if sieve-parser
    (sieve-register-test "mimeheader"
			 test-mimeheader
			 (list 'string-list 'string-list)
			 (append comparator match-type)))
