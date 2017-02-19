;;;; GNU Mailutils -- a suite of utilities for electronic mail
;;;; Copyright (C) 1999, 2000, 2001, 2006, 2007, 2010 Free Software
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

;;;; This module provides sieve's "redirect" action.

;;; rfc3028 says: 
;;; "Implementations SHOULD take measures to implement loop control,"
;;; We do this by appending an "X-Sender" header to each message
;;; being redirected. If one of the "X-Sender" headers of the message
;;; contains our email address, we assume it is a loop and bail out.

(define (sent-from-me? msg)
  (call-with-current-continuation
   (lambda (exit)
     (for-each
      (lambda (hdr)
	(if (and (string-ci=? (car hdr) "X-Sender")
		 (string-ci=? (mu-address-get-email (cdr hdr))
			   sieve-my-email))
	    (exit #t))) 
      (mu-message-get-header-fields sieve-current-message))
     #f)))

;;; redirect action
(define (action-redirect address)
  (sieve-verbose-print "REDIRECT" "to address " address)
  (handle-exception
   (if sieve-my-email
       (cond
	((sent-from-me? sieve-current-message)
	 (runtime-message SIEVE-WARNING "Redirection loop detected"))
	(else
	 (let ((out-msg (mu-message-copy sieve-current-message))
	       (sender (mu-message-get-sender sieve-current-message)))
	   (mu-message-set-header out-msg "X-Sender" sieve-my-email)
	   (mu-message-send out-msg #f sender address)
	   (mu-message-destroy out-msg))
	 (mu-message-delete sieve-current-message))))))

;;; Register action
(if sieve-parser
    (sieve-register-action "redirect" action-redirect (list 'string) '()))



