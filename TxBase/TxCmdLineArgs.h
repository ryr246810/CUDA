//--------------------------------------------------------------------
//
// File:    TxCmdLineArgs.h
//
// Purpose: Holder of the arguments on a command line
//
// Version: $Id: TxCmdLineArgs.h 58 2006-09-18 17:25:05Z yew $
//
// Tech-X Corporation, 2000
//
//--------------------------------------------------------------------

#ifndef TX_CMD_LINE_ARGS_H
#define TX_CMD_LINE_ARGS_H

// system includes
#include "TxAttributeSet.h"

/**
 * A TxCmdLineArgs parses out the command line data from the command line
 * arguments.  If it can convert a variable to an int, it appends an
 * option.  If it can convert to a double, it appends a parameter.  Otherwise
 * it appends a string.
 */
class TxCmdLineArgs : public TxAttributeSet
{
public:
    /**
     * Default constructor
     */
    TxCmdLineArgs(std::string s) : TxAttributeSet(s) {}

    /**
     * Destructor
     */
    virtual ~TxCmdLineArgs() {}

    /**
     * Assignment from the argc and argv of a command line
     */
    virtual void setFromCmdLine(int argc, char **argv);

private:
    // To prevent use
    TxCmdLineArgs(const TxCmdLineArgs &);
    TxCmdLineArgs &operator=(const TxCmdLineArgs &);
};

#endif // TX_CMD_LINE_ARGS_H
