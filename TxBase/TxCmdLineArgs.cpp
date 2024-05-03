//--------------------------------------------------------------------
//
// File:    TxCmdLineArgs.cpp
//
// Purpose: Holder of command line args as an attribute set
//
// Version: $Id: TxCmdLineArgs.cpp 85 2007-01-19 18:42:35Z sizemore $
//
// Tech-X Corporation, 2000
//
//--------------------------------------------------------------------

// Standard includes

#include "TxStreams.h"
#include <stdlib.h>
// system includes
#include "TxCmdLineArgs.h"

// Assignment from the argc and argv of a command line
void TxCmdLineArgs::setFromCmdLine(int argc, char** argv) {

// Read each variable.  If an option or param, so convert.
// Otherwise store as a string
  setObjectName(argv[0]);
  for (int i=1; i<argc; ++i) {
    if ( argv[i][0] != '-' ) {
      cout << "Command line argument " << argv[i] << " ignored.\n";
      continue;
    }
    char* name = &argv[i][1];
    if (i < argc-1) {
      if ( argv[i+1][0] == '-' ) {
        if (hasOption(name)) setOption(name, 1);
        else cout << "Command line argument " << argv[i] << " ignored.\n";
      }
      else{
        if (hasOption(name)) {
	  int opt = atoi(argv[i+1]);
	  setOption(name, opt);
	}
        else if (hasParam(name)) {
	  double prm = atof(argv[i+1]);
	  setParam(name, prm);
	}
        else if (hasString(name)) setString(name, argv[i+1]);
        else cout << "Command line argument " << argv[i] << " ignored.\n";
        ++i;
      }
    }
    else{
      if (hasOption(name)) setOption(name, 1);
      else cout << "Command line argument " << name << "ignored.\n";
    }

  }

}
