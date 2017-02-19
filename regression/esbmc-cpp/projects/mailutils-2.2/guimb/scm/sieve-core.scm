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

;;;; This module provides core functionality for the sieve scripts.

(define-module (mailutils sieve-core))

(use-modules (mailutils mailutils))

;;; Set to #t when parsing 
(define-public sieve-parser #f)

;;; Name of the input source
(define-public sieve-source "UNKNOWN")

;;; The email address for originator of error messages. Should be <>
;;; but current mailutils API is unable to parse and handle it.
;;; Site administrators are supposed to replace it with the
;;; actual value.
(define-public sieve-daemon-email "MAILER-DAEMON@localhost")

;;; The email address of the user whose mailbox is being processed.
;;; If #f, it will be set by sieve-main
(define-public sieve-my-email #f)

(define SIEVE-WARNING "Warning")
(define SIEVE-ERROR "Error")
(define SIEVE-NOTICE "Notice")

(defmacro handle-exception (. expr)
  `(catch 'mailutils-error
	  (lambda () ,@expr)
	  (lambda (key . args)
	    (runtime-message SIEVE-ERROR
			     "In function " (car args) ": "
			     (apply format #f
				    (list-ref args 1) (list-ref args 2))
			     (let ((error-code
				    (car (list-ref args (1- (length args))))))
			       (if (= error-code 0)
				   ""
				   (string-append
				    "; Error code: "
				    (number->string error-code)
				    " - "
				    (mu-strerror error-code))))))))

;;; Set to #t if verbose action listing is requested
(define-public sieve-verbose #f)

(defmacro sieve-verbose-print (action . rest)
  `(if sieve-verbose
       (let ((uid (false-if-exception
		   (mu-message-get-uid sieve-current-message))))
	 (display ,action)
	 (display " on msg uid ")
	 (display uid)
	 (let ((args (list ,@rest)))
	   (cond ((not (null? args))
		  (display ": ")
		  (for-each
		   display
		   args))))
	 (newline))))

;;; List of open mailboxes.
;;; Each entry is: (list MAILBOX-NAME OPEN-FLAGS MBOX)
(define sieve-mailbox-list '())

;;; Cached mailbox open: Lookup in the list first, if not found,
;;; call mu-mailbox-open and append to the list.
;;; NOTE: second element of each slot (OPEN-FLAGS) is not currently
;;; used, sinse all the mailboxes are open with "cw".
(define (sieve-mailbox-open name flags)
  (let ((slot (assoc name sieve-mailbox-list)))
    (if slot
	(list-ref slot 2)
      (let ((mbox (false-if-exception (mu-mailbox-open name flags))))
	(if mbox
	    (set! sieve-mailbox-list (append
				      sieve-mailbox-list
				      (list
				       (list name flags mbox)))))
	mbox))))

;;; Close all open mailboxes.
(define (sieve-close-mailboxes)
  (for-each
   (lambda (slot)
     (cond
      ((list-ref slot 2)
       => (lambda (mbox)
	    (mu-mailbox-close mbox)))))
   sieve-mailbox-list)
  (set! sieve-mailbox-list '()))

(define (sieve-expand-filename filename)
  (case (string-ref filename 0)
    ((#\/ #\% #\~ #\+ #\=)
      filename)
    (else
     (let ((pw (getpwuid (geteuid))))
	(if (vector? pw)
	    (string-append (vector-ref pw 5)
			   "/"
			   filename)
	    filename)))))

;;; Comparators
(define-public sieve-standard-comparators 
  (list (list "i;octet" string=?)
	(list "i;ascii-casemap" string-ci=?)))

;;; Stop statement

(define-public (sieve-stop)
  (sieve-verbose-print "STOP")
  (throw 'sieve-stop))

;;; Basic five actions:

;;; reject is defined in reject.scm

;;; fileinto

(define-public (action-fileinto filename)
  (let ((name (sieve-expand-filename filename)))
    (sieve-verbose-print "FILEINTO" "delivering into " name)
    (if (string? name)
	(let ((outbox (sieve-mailbox-open name "cw")))
	  (cond
	   (outbox
	    (handle-exception
	     (mu-mailbox-append-message outbox sieve-current-message)
	     (mu-message-delete sieve-current-message)))
	   (else
	    (runtime-message SIEVE-ERROR
			     "Could not open mailbox " name))))
	(runtime-message SIEVE-ERROR
			 "Could not expand mailbox name " filename))))

;;; redirect is defined in redirect.scm

;;; keep -- does nothing worth mentioning :^)

(define-public (action-keep)
  (sieve-verbose-print "KEEP")
  (handle-exception
   (mu-message-delete sieve-current-message #f)))

;;; discard

(define-public (action-discard)
  (sieve-verbose-print "DISCARD" "marking as deleted")
  (handle-exception
   (mu-message-delete sieve-current-message)))

;;; Register standard actions
(define-public sieve-standard-actions
  (list (list "keep" action-keep '() '())
	(list "discard" action-discard '() '())
	(list "fileinto" action-fileinto (list 'string) '())))

;;; Some utilities.

(define (sieve-get-opt-arg opt-args tag default)
  (cond
   ((member tag opt-args) =>
    (lambda (x)
      (car (cdr x))))
   (else
    default)))

(define (find-comp opt-args)
  (sieve-get-opt-arg opt-args #:comparator string-ci=?))

(define (find-match opt-args)
  (cond
   ((member #:is opt-args)
    #:is)
   ((member #:contains opt-args)
    #:contains)
   ((member #:matches opt-args)
    #:matches)
   ((member #:regex opt-args)
    #:regex)
   (else
    #:is)))

(define (sieve-str-str str key comp)
  (if (string-null? key)
      ;; rfc3028:
      ;;   If a header listed in the header-names argument exists, it contains
      ;;   the null key ("").  However, if the named header is not present, it
      ;;   does not contain the null key.
      ;; This function gets called only if the header was present. So:
      #t
    (let* ((char (string-ref key 0))
	   (str-len (string-length str))
	   (key-len (string-length key))
	   (limit (- str-len key-len)))
      (if (< limit 0)
	  #f
	(call-with-current-continuation
	 (lambda (xx)
	   (do ((index 0 (1+ index)))
	       ((cond
		 ((> index limit)
		  #t)
		 ;; FIXME: This is very inefficient, but I have to use this
		 ;; provided (string-index str (string-ref key 0)) may not
		 ;; work...
		 ((comp (substring str index (+ index key-len))
			key)
		  (xx #t))
		 (else
		  #f)) #f))
	   #f))))))

;;; Convert sieve-style regexps to POSIX:

(define (sieve-regexp-to-posix regexp)
  (let ((length (string-length regexp)))
    (do ((cl '())
	 (escape #f)
	 (i 0 (1+ i)))
	((= i length) (list->string (reverse cl)))
      (let ((ch (string-ref regexp i)))
	(cond
	 (escape
	  (set! cl (append (list ch) cl))
	  (set! escape #f))
	 ((char=? ch #\\)
	  (set! escape #t))
	 ((char=? ch #\?)
	  (set! cl (append (list #\.) cl)))
	 ((char=? ch #\*)
	  (set! cl (append (list #\* #\.) cl)))
	 ((member ch (list #\. #\$ #\^ #\[ #\]))
	  (set! cl (append (list ch #\\) cl)))
	 (else
	  (set! cl (append (list ch) cl))))))))


(define (get-regex match key comp)
  (case match
    ((#:matches)
     (make-regexp (sieve-regexp-to-posix key)
		  (if (eq? comp string-ci=?)
		      regexp/icase
		      '())))
    ((#:regex)
     (make-regexp key
		  (if (eq? comp string-ci=?)
		      regexp/icase
		      '())))
    (else
     #f)))

;;;; Standard tests:

(define-public (test-address header-list key-list . opt-args)
  (let ((comp (find-comp opt-args))
	(match (find-match opt-args))
	(part (cond
	       ((member #:localpart opt-args)
		#:localpart)
	       ((member #:domain opt-args)
		#:domain)
	       (else
		#:all))))
    (call-with-current-continuation
     (lambda (exit)
       (for-each
	(lambda (key)
	  (let ((header-fields (mu-message-get-header-fields
				sieve-current-message
				header-list))
		(rx (get-regex match key comp)))
	    (for-each
	     (lambda (h)
	       (let ((hdr (cdr h)))
		 (if hdr
		     (let ((naddr (mu-address-get-count hdr)))
		       (do ((n 1 (1+ n)))
			   ((> n naddr) #f)
			 (let ((addr (case part
				       ((#:all)
					(mu-address-get-email hdr n))
				       ((#:localpart)
					(mu-address-get-local hdr n))
				       ((#:domain)
					(mu-address-get-domain hdr n)))))
			   (if addr
			       (case match
				 ((#:is)
				  (if (comp addr key)
				      (exit #t)))
				 ((#:contains)
				  (if (sieve-str-str addr key comp)
				      (exit #t)))
				 ((#:matches #:regex)
				  (if (regexp-exec rx addr)
				      (exit #t))))
			       (runtime-message SIEVE-NOTICE
				"Can't get address parts for message "
				sieve-current-message))))))))
	     header-fields)))
	key-list)
       #f))))

(define-public (test-size key-size . comp)
  (let ((size (mu-message-get-size sieve-current-message)))
    (cond
     ((null? comp)       ;; An extension.
      (= size key-size)) 
     ((eq? (car comp) #:over)
      (> size key-size))
     ((eq? (car comp) #:under)
      (< size key-size))
     (else
      (runtime-message SIEVE-ERROR  "test-size: unknown comparator " comp)))))

(define-public (test-envelope part-list key-list . opt-args)
  (let ((comp (find-comp opt-args))
	(match (find-match opt-args)))
    (call-with-current-continuation
     (lambda (exit)
       (for-each
	(lambda (part)
	  (cond
	   ((string-ci=? part "From")
	    (let ((sender (mu-message-get-sender sieve-current-message)))
	      (for-each
	       (lambda (key)
		 (if (comp key sender)
		     (exit #t)))
	       key-list)))
	   (else
	    (runtime-message SIEVE-ERROR
			     "Envelope part " part " not supported")
	    #f)))
	part-list)
       #f))))

(define-public (test-exists header-list)
  (call-with-current-continuation
   (lambda (exit)
     (for-each (lambda (hdr)
		 (let ((val (mu-message-get-header sieve-current-message hdr)))
                   (if (or (not val) (= (string-length val) 0))
		     (exit #f))))
	       header-list)
     #t)))

(define-public (test-header header-list key-list . opt-args)
  (let ((comp (find-comp opt-args))
	(match (find-match opt-args)))
    (call-with-current-continuation
     (lambda (exit)
       (for-each
	(lambda (key)
	  (let ((header-fields (mu-message-get-header-fields
				sieve-current-message
				header-list))
		(rx (get-regex match key comp)))
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
		       ((#:matches #:regex)
			(if (regexp-exec rx hdr)
			    (exit #t)))))))
	     header-fields)))
	key-list)
       #f))))

;;; Register tests:
(define address-part (list (cons "localpart" #f)
               		   (cons "domain" #f)
			   (cons "all" #f)))
(define match-type   (list (cons "is" #f)
			   (cons "contains" #f)
			   (cons "matches" #f)
			   (cons "regex" #f)))
(define size-comp    (list (cons "under" #f)
			   (cons "over" #f)))
(define comparator   (list (cons "comparator" 'string)))

(define-public sieve-standard-tests
  (list
   (list "address"
	 test-address
	 (list 'string-list 'string-list)
	 (append address-part comparator match-type))
   (list "size"
	 test-size
	 (list 'number)
	 size-comp)
   (list "envelope"
	 test-envelope
	 (list 'string-list 'string-list)
	 (append comparator address-part match-type))
   (list "exists"
	 test-exists
	 (list 'string-list) 
	 '())
   (list "header"
	 test-header
	 (list 'string-list 'string-list)
	 (append comparator match-type))
   (list "false" #f '() '())
   (list "true"  #t '() '())))

;;; runtime-message

(define-public (runtime-message level . text)
  (let ((msg (apply string-append
		    (map (lambda (x)
			   (format #f "~A" x))
			 (append
			  (list "(in " sieve-source ") ")
			  text)))))
    (if sieve-current-message
	(mu-message-set-header sieve-current-message
			       (string-append "X-Sieve-" level)
			       msg))
    (if (isatty? (current-error-port))
	(display (string-append level ": " msg "\n") (current-error-port)))))

(define (guimb?)
  (catch #t
	 (lambda ()
	   (let ((v current-mailbox))
	     v))
	 (lambda args #f)))

;;; Sieve-main
(define-public sieve-mailbox #f)
(define-public sieve-current-message #f)

(define-public (sieve-run-current-message thunk)
  (and (catch 'sieve-stop
	      thunk
	      (lambda args
		#f))
       (sieve-verbose-print "IMPLICIT KEEP")))

(define (sieve-run thunk)
  (if (not sieve-my-email)
      (set! sieve-my-email (mu-username->email)))
;  (DEBUG 1 "Mailbox: " sieve-mailbox)

  (let msg-loop ((msg (mu-mailbox-first-message sieve-mailbox)))
    (if (not (eof-object? msg))
	(begin
	  (set! sieve-current-message msg)
	  (sieve-run-current-message thunk)
	  (msg-loop (mu-mailbox-next-message sieve-mailbox)))))

  (sieve-close-mailboxes))

(define (sieve-command-line)
  (catch #t
	 (lambda ()
	   (let ((args sieve-script-args))
	     (append (list "<temp-file>") args)))
	 (lambda args (command-line))))

(define-public (sieve-main thunk) 
  (handle-exception
   (cond
    ((not (guimb?))
     (let* ((cl (sieve-command-line))
	    (name (if (and (not (null? (cdr cl)))
			   (string? (cadr cl)))
		      (cadr cl)
		      (mu-user-mailbox-url
		       (passwd:name (mu-getpwuid (getuid)))))))

       (set! sieve-mailbox (mu-mailbox-open name "rw"))
       (sieve-run thunk)
       (mu-mailbox-expunge sieve-mailbox)
       (mu-mailbox-close sieve-mailbox)))
    (else
     (set! sieve-mailbox current-mailbox)
     (sieve-run thunk)))))
	 
