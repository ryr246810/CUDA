#ifndef _RunCmdLineArgs_HeaderFile
#define _RunCmdLineArgs_HeaderFile

// system includes
#include <TxCmdLineArgs.h>

class RunCmdLineArgs : public TxCmdLineArgs
{

public:
  RunCmdLineArgs();

  virtual ~RunCmdLineArgs(){}
  
private:
  // To prevent use
  RunCmdLineArgs(const RunCmdLineArgs&);
  RunCmdLineArgs& operator=(const RunCmdLineArgs&);
};

#endif	// _RunCmdLineArgs_HeaderFile

