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

;;; This is a simple Guile program that generates automatic reply to 
;;; incoming mail messages.
;;;
;;; usage: to your /etc/aliases add:
;;;
;;;   username: "|/usr/local/bin/guimb -f <path>/reply.scm"
;;;
;;; and adjust variables below to your liking.
;;; Any message to the address username@your.host will be responded
;;; and (optionally) saved in a mailbox.

(define indent-prefix "> ")
(define save-mailbox #f)
(define reply-text
  "Sorry, I am not here to attend your message. I will do\n\
it as soon as I come back.\n\n\
Kind regards\n")

;; Reply to the incoming message
(define (reply in-msg)
  (let* ((out-msg (mu-message-create))
	 (in-port (mu-message-get-port in-msg "r"))
	 (out-port (mu-message-get-port out-msg "w")))
    (mu-message-set-header out-msg "To"
			   (mu-message-get-header in-msg "From"))
    (mu-message-set-header out-msg "Cc"
			   (mu-message-get-header in-msg "Cc"))
    (mu-message-set-header out-msg "Subject"
			   (string-append
			    "Re: "
			    (mu-message-get-header in-msg "Subject")))

    (display reply-text out-port)
    
    (display "\n\nOriginal message:\n" out-port)
    (do ((hdr (mu-message-get-header-fields in-msg) (cdr hdr)))
	 ((null? hdr) #f)
	(let ((s (car hdr)))
	  (display (string-append
		    indent-prefix
		    (car s) ": " (cdr s) "\n") out-port)))
    (display (string-append indent-prefix "\n") out-port)
    (do ((line (read-line in-port) (read-line in-port)))
	((eof-object? line) #f)
      (display (string-append indent-prefix line "\n") out-port))

    (close-input-port in-port)
    (close-output-port out-port)
 	
    (mu-message-send out-msg)))

;;; Upon receiving a message, store it away in the save mailbox and
;;; reply to the sender.
(let ((mbox (and save-mailbox (mu-mailbox-open save-mailbox "cw")))
      (msg (mu-mailbox-get-message current-mailbox 1)))
  (cond
   (mbox
    (mu-mailbox-append-message mbox msg)
    (mu-mailbox-close mbox)))
  (reply msg))	
    
    
	
