//--------------------------------------------------------------------
// Purpose:	Holder of command line args as an attribute set

// txbase includes
#include <TxStreams.h>

// 
#include <RunCmdLineArgs.hxx>

// Constructor: set the known command line options
RunCmdLineArgs::RunCmdLineArgs() : TxCmdLineArgs("RunCmdLineArgs")
{
  // The work dir
  appendString("wd", "");

  appendOption("dp", 0);  

  appendOption("fdp", 0);  

  appendOption("pdp", 0);  

  appendOption("n", 1);  
}

