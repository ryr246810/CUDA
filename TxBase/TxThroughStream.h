//-----------------------------------------------------------------------
//
// File:    TxThroughStream.h
//
// Purpose: Class for connecting multiple ostreams through a single ostream.
//
// Version: $Id: TxThroughStream.h 61 2006-09-18 18:51:45Z yew $
//
// Copyright (c) 1999 by Tech-X Corporation.  All rights reserved.
//
//-----------------------------------------------------------------------


#ifndef TX_THROUGH_STREAM_H
#define TX_THROUGH_STREAM_H

#ifdef _WIN32
// #pragma warning ( disable: 4786)
// innocent pragma to keep the VC++ compiler quiet
// about the too long names in STL headers.
#endif

// std includes
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

// txbase includes
#include <TxBase.h>
// The following prevents multiple instantiation of
// the streambuf class
// JRC: Still needed for cxx even though cxx -M complains.
#if defined(__DECCXX)
// #pragma do_not_instantiate basic_streambuf<char>
#endif

/** Class for connecting multiple ostreams through a single ostream.
 *
 *  A TxThroughStream is an ostream for which the output can be connected
 *  to multiple other ostreams.  Output to a TxThroughStream can be labeled
 *  with one or more filter flags specifing the nature of the output.  Ostreams
 *  attached to the TxThroughStream can register to selectively listen to
 *  given types of output.
 *
 *  Copyright (c) 1999 by Tech-X Corporation.  All rights reserved.
 *
 *  @author  Ryan McLean, Kelly G. Luetkemeyer
 *  @version $Id: TxThroughStream.h 61 2006-09-18 18:51:45Z yew $
 */
class TXBASE_API TxThroughStream : public std::streambuf, public std::ostream
{

  public:
/**
 * These are the types of ways in which output can be labeled.
 * These filter types can be applied to a TxThroughStream via the
 * setFilter method, in which case they label the type of all future
 * output until another filter type is applied.  Or the filter types
 * can be used to describe the types of output that an attached ostream
 * listens to.  The names are descriptive, but will only have meaning
 * when conventions are established.  Filters can be combined in the
 * same way that flags can.
 *
 * Types:
 * OFF -- No output; listen to nothing.
 * OUTPUT, MESG, DEBUG, WARNING, ALERT, ERRORS, USER -- Meaning established through use
 * ALL -- Send to all streams; listen to everything.
 *
 */
  enum txfilter { TX_OFF = 0,   TX_OUTPUT=1,  TX_MESG=2,
                  TX_DEBUG=4,   TX_WARNING=8, TX_ALERT=16,
                  TX_ERRORS=32, TX_USER=64,   TX_ALL=127};

/**
 * The TxThroughStream defaults to sending output to all attached
 * ostreams.
 */
    TxThroughStream();

/**
 * This TxThroughStream sends output to the input parameter ostream
 * @param os the desired ostream to attach
 * @param f the desired filter -- defaults to TX_ALL
 */
    TxThroughStream(std::ostream &os, txfilter f = TX_ALL);

/**
 * Virtual destructor flushes remaining output.
 */
    virtual ~TxThroughStream();

/**
 * Attaches the stream os to the TxThroughStream with filter type
 * f.  Any output sent to the TxThroughStream that matches one of
 * the types in f will be passed on to s.  Calling attachStream
 * with os already attached will reset the filter type to f.
 *
 * @param os the ostream to attach.
 * @param f the types of messages that os will listen to.  Defaults
 * to ALL.
 */
    void attachStream(std::ostream &os, txfilter f = TX_ALL) {
      strStateMap[&os] = f;
    }

/**
 * Disconnects os from the TxThroughStream.  Functionaly equivalent to
 * attachStream(os, OFF).
 *
 * @param os the ostream to detach
 */
    void detachStream(std::ostream &os);

/**
 * Labels the type for suceeding output to the TxThroughStream.
 * Output has type f until this function is called again.
 *
 * @param f the type of future messages to TxThroughStream.
 */
    void setFilter(txfilter f) {
      state = f;
    }

/**
 * Print time: put the time into the through stream.
 *
 * @return reference to this stream for further printing
 */
    TxThroughStream& printTime();

  protected:

/// sync method from streambuf.
    virtual int sync ();
/**
 * Helper function to output length characters from text.
 * Output is sent to attached ostreams if they the types match.
 *
 * @param text the text buffer
 * @param length the number of characters to output
 * @returns the number of characters sucessfully output.
 */
    virtual int output (char *text, int length);

/// Associate array of attached streams.
    std::map<std::ostream*, txfilter> strStateMap;

/// Type of current output.
    txfilter state;

  private:
/// Do not allow copying.
    TxThroughStream(const TxThroughStream &);
/// Do not allow assignment.
    TxThroughStream& operator=(const TxThroughStream &);

// Either trusted flags or RTTI is required to connect
// one TxThroughStream to another with the state handled
// properly.
//std::map<ostrPtr, bool> txThroughStreamConnection;

/// Use overflow to do output.  From streambuf base class.
    int overflow(int ch);

};

#endif
