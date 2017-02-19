/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2005, 2007, 2010 Free Software Foundation,
   Inc.

   GNU Mailutils is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   GNU Mailutils is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GNU Mailutils; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
   MA 02110-1301 USA */

#include "mail.h"

#define FIND_IN_TABLE(f,t,c)\
  f ((void*)(t), sizeof ((t)) / sizeof ((t)[0]), sizeof ((t)[0]), c)

static const struct mail_command_entry mail_command_table[] = {
  { "a",	"alias",	"a[lias] [alias [address...]]",	0,
    mail_alias, alias_compl },
  { "alt",	"alternates",	"alt[ernates] name...",		0,
    mail_alt, NULL },
  { "C",	"Copy",		"C[opy] [msglist]",		0,
    mail_copy,  msglist_compl },
  { "cd",	"cd",		"cd [directory]",		0,
    mail_cd, dir_compl },
  { "ch",	"chdir",	"ch[dir] directory",		0,
    mail_cd, NULL },
  { "c",	"copy",		"c[opy] [[msglist] file]",	0,
    mail_copy, msglist_file_compl },
  { "dec",	"decode",	"dec[ode] [msglist]",		0,
    mail_decode, msglist_compl },
  { "d",	"delete",	"d[elete] [msglist]",		0,
    mail_delete, msglist_compl },
  { "di",	"discard",	"di[scard] [header-field...]",	0,
    mail_discard, no_compl },
  { "dp",	"dp",		"dp [msglist]",			0,
    mail_dp, msglist_compl },
  { "dt",	"dt",		"dt [msglist]",			0,
    mail_dp, msglist_compl },
  { "ec",	"echo",		"ec[ho] string ...",		0,
    mail_echo, no_compl },
  { "e",	"edit",		"e[dit] [msglist]",		0,
    mail_edit, msglist_compl },
  { "el",	"else",		"el[se]",			EF_FLOW,
    mail_else, no_compl },
  { "en",	"endif",	"en[dif]",			EF_FLOW,
    mail_endif, no_compl },
  { "env",      "envelope",     "en[velope]",                   0,
    mail_envelope, msglist_compl },
  { "ex",	"exit",		"ex[it]",			0,
    mail_exit, no_compl },
  { "F",	"Followup",	"F[ollowup] [msglist]",		EF_SEND,
    mail_followup, msglist_compl },
  { "fi",	"file",		"fi[le] [file]",		0,
    mail_file, file_compl },
  { "fold",	"folder",	"fold[er] [file]",		0,
    mail_file, file_compl },
  { "folders",	"folders",	"folders",			0,
    mail_folders, no_compl },
  { "fo",	"followup",	"fo[llowup] [msglist]",		EF_SEND,
    mail_followup, msglist_compl },
  { "f",	"from",		"f[rom] [msglist]",		0,
    mail_from, msglist_compl },
  { "g",	"group",	"g[roup] [alias [address...]]",	0,
    mail_alias, alias_compl },
  { "h",	"headers",	"h[eaders] [msglist]",		0,
    mail_headers, msglist_compl },
  { "hel",	"help",		"hel[p] [command...]",		0,
    mail_help, command_compl },
  { "ho",	"hold",		"ho[ld] [msglist]",		0,
    mail_hold, msglist_compl },
  { "i",	"if",		"i[f] s|r|t",			EF_FLOW,
    mail_if, no_compl },
  { "ig",	"ignore",	"ig[nore] [header-field...]",	0,
    mail_discard, no_compl },
  { "inc",      "incorporate",	"inc[orporate]",		0,
    mail_inc, no_compl },
  { "l",	"list",		"l[ist]",			0,
    mail_list, no_compl },
  { "m",	"mail",		"m[ail] [address...]",		EF_SEND,
    mail_send, alias_compl },
  { "mb",	"mbox",		"mb[ox] [msglist]",		0,
    mail_mbox, msglist_compl },
  { "n",	"next",		"n[ext] [message]",		0,
    mail_next, no_compl },
  { "nosender", "nos",          "nos[ender] [header-field...]", 0,
    mail_nosender, no_compl },
  { "nou",      "nounfold",     "nou[nfold] [header-field]",    0,
    mail_nounfold, no_compl },
  { "P",	"Print",	"P[rint] [msglist]",		0,
    mail_print, msglist_compl },
  { "pi",	"pipe",		"pi[pe] [[msglist] command]",	0,
    mail_pipe, no_compl }, /* FIXME: exec_compl */
  { "pre",	"preserve",	"pre[serve] [msglist]",		0,
    mail_hold, msglist_compl },
  { "prev",	"previous",	"prev[ious] [message]",		0,
    mail_previous, no_compl },
  { "p",	"print",	"p[rint] [msglist]",		0,
    mail_print, msglist_compl },
  { "q",	"quit",		"q[uit]",			0,
    mail_quit, no_compl },
  { "R",	"Reply",	"R[eply] [msglist]",		EF_SEND,
    mail_reply, msglist_compl },
  { "R",	"Respond",	"R[espond] [msglist]",		EF_SEND,
    mail_reply, msglist_compl },
  { "r",	"reply",	"r[eply] [msglist]",		EF_SEND,
    mail_reply, msglist_compl },
  { "r",	"respond",	"r[espond] [msglist]",		EF_SEND,
    mail_reply, msglist_compl },
  { "ret",	"retain",	"ret[ain] [header-field]",	0,
    mail_retain, no_compl },
  { "S",	"Save",		"S[ave] [msglist]",		0,
    mail_save, msglist_compl },
  { "s",	"save",		"s[ave] [[msglist] file]",	0,
    mail_save, msglist_file_compl },
  { "sen",      "sendheader",   "sen[dheader] [[header][: value]]", EF_SEND,
    mail_sendheader, no_compl },
  { "se", "set", "se[t] [name[=[string]]...] [name=number...] [noname...]", 0,
    mail_set, mailvar_set_compl },
  { "setq",     "setq",         NULL,                           EF_HIDDEN,
    mail_set, no_compl },
  { "sender",   "sen",          "sen[der] [header-field...]",   0,
    mail_sender, no_compl },
  { "sete",     "setenv",       "sete[nv] [name[=value]]",      0,
    mail_setenv, no_compl },
  { "sh",	"shell",	"sh[ell] [command]",		0,
    mail_shell, no_compl }, /* FIXME: exec_compl */
  { "si",	"size",		"si[ze] [msglist]",		0,
    mail_size, msglist_compl },
  { "so",	"source",	"so[urce] file",		0,
    mail_source, NULL },
  { "st",       "struct",       "st[ruct] [msglist]",           0,
    mail_struct, NULL },
  { "su",	"summary",	"su[mmary]",			0,
    mail_summary, no_compl },
  { "T",	"Type",		"T[ype] [msglist]",		0,
    mail_print, msglist_compl },
  { "ta",       "tag",		"ta[g] [msglist]",		0,
    mail_tag, msglist_compl },
  { "to",	"top",		"to[p] [msglist]",		0,
    mail_top, msglist_compl },
  { "tou",	"touch",        "tou[ch] [msglist]",  	        0,
    mail_touch, msglist_compl },
  { "t",	"type",		"t[ype] [msglist]",		0,
    mail_print, msglist_compl },
  { "una",	"unalias",	"una[lias] [alias]...",		0,
    mail_unalias, NULL },
  { "u",	"undelete",	"u[ndelete] [msglist]",		0,
    mail_undelete, msglist_compl },
  { "unf",      "unfold",       "unf[old] [header-field]",      0,
    mail_unfold, no_compl },
  { "uns",	"unset",	"uns[et] name...",		0,
    mail_unset, mailvar_set_compl },
  { "unt",      "untag",	"unt[ag] [msglist]",		0,
    mail_tag, msglist_compl },
  { "va",       "variable",     "variable [name...]",           0,
    mail_variable, mailvar_set_compl },
  { "ve",	"version",	"ve[rsion]",			0,
    mail_version, no_compl },
  { "v",	"visual",	"v[isual] [msglist]",		0,
    mail_visual, msglist_compl },
  { "wa",       "warranty",	"wa[rranty]",			0,
    mail_warranty, no_compl },
  { "W",	"Write",	"W[rite] [msglist]",		0,
    mail_write, msglist_compl },
  { "w",	"write",	"w[rite] [[msglist] file]",	0,
    mail_write, msglist_file_compl },
  { "x",	"xit",		"x[it]",			0,
    mail_exit, no_compl },
  { "z",	"",		"z[+|-|. [count]]",		0,
    mail_z, no_compl },
  { "?",	"?",		"? [command...]",		0,
    mail_help, command_compl },
  { "!",	"",		"![command]",			0,
    mail_shell, exec_compl },
  { "=",	"=",		"=",				0,
    mail_eq, no_compl },
  { "#",	"#",		"# comment",			0,
    NULL, no_compl },
  { "*",	"*",		"*",				0,
    mail_list, no_compl },
  { "+",	"+",		"+ [message]",			0,
    mail_next, msglist_compl },
  { "|",	"|",		"| [[msglist] command]",	0,
    mail_pipe, msglist_compl }, /* FIXME: msglist_exec_compl */
  { "-",	"-",		"- [message]",			0,
    mail_previous, msglist_compl },
};


const struct mail_command_entry *
mail_find_command (const char *cmd)
{
  return FIND_IN_TABLE (util_find_entry, mail_command_table, cmd);
}

int
mail_command_help (const char *cmd)
{
  return FIND_IN_TABLE (util_help, mail_command_table, cmd);
}

void
mail_command_list ()
{
  util_command_list ((void*)mail_command_table,
		     sizeof (mail_command_table) / sizeof (mail_command_table[0]),
		     sizeof (mail_command_table[0]));
}

const struct mail_command *
mail_command_name (int i)
{
  if (i < 0 || i >= sizeof (mail_command_table) / sizeof (mail_command_table[0]))
    return NULL;
  return (struct mail_command*) &mail_command_table[i];
}

static const struct mail_escape_entry mail_escape_table[] = {
  {"!",	"!",	"![shell-command]", escape_shell },
  {":",	":",	":[mail-command]",  escape_command },
  {"-",	"-",	"-[mail-command]",  escape_command },
  {"?",	"?",	"?",		    escape_help },
  {"A",	"A",	"A",		    escape_sign },
  {"a",	"a",	"a",		    escape_sign },
  {"b",	"b",	"b[bcc-list]",	    escape_bcc },
  {"c",	"c",	"c[cc-list]",	    escape_cc },
  {"d",	"d",	"d",		    escape_deadletter },
  {"e",	"e",	"e",		    escape_editor },
  {"f",	"f",	"f[mesg-list]",	    escape_print },
  {"F",	"F",	"F[mesg-list]",	    escape_print },
  {"h",	"h",	"h",		    escape_headers },
  {"i",	"i",	"i[var-name]",	    escape_insert },
  {"m",	"m",	"m[mesg-list]",	    escape_quote },
  {"M",	"M",	"M[mesg-list]",	    escape_quote },
  {"p",	"p",	"p",		    escape_type_input },
  {"r",	"<",	"r[filename]",	    escape_read },
  {"s",	"s",	"s[string]",	    escape_subj },
  {"t",	"t",	"t[name-list]",	    escape_to },
  {"v",	"v",	"v",		    escape_visual },
  {"w",	"w",	"w[filename]",	    escape_write },
  {"x", "x",    "x",                NULL }, /* Implemented directly in
					       send.c */
  {"|",	"|",	"|[shell-command]", escape_pipe },
};

const struct mail_escape_entry *
mail_find_escape (const char *cmd)
{
  return FIND_IN_TABLE (util_find_entry, mail_escape_table, cmd);
}

int
mail_escape_help (const char *cmd)
{
  return FIND_IN_TABLE (util_help, mail_escape_table, cmd);
}

