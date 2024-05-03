
#ifndef _DynObj_HeaderFile
#define _DynObj_HeaderFile

// std includes
#include <string>

// txbase includes
#include <TxStreams.h>
#include <TxThroughStream.h>


#include <DynObjBase.hxx>


/*** A DynObj is a dynamic object that has some implementation of time step saving. **/

class DynObj : public DynObjBase
{
public:

  /*** Constructor */
  DynObj() : DynObjBase() {
    m_ThruStrmPtr = new TxThroughStream(std::cout);
    m_ThruStrmOwner = true;
    m_CurTime = 0.;
    m_DumpCntr = 0;
    m_DumpPeriod = 1;
  };

  /*** Constructor **/
  DynObj(std::string nm) : DynObjBase(nm) {
    m_ThruStrmPtr = new TxThroughStream(std::cout);
    m_ThruStrmOwner = true;
    m_CurTime = 0.;
    m_DumpCntr = 0;
    m_DumpPeriod = 1;
  }

  /*** Destructor */
  virtual ~DynObj(){
    if(m_ThruStrmOwner) delete m_ThruStrmPtr;
  };
  
  /**
   * Set the stream for comments.  This is not owned.
   *
   * @param cs the stream to write comments to
   */
  virtual void SetThruStrm(TxThroughStream& cs){
    if(m_ThruStrmOwner) delete m_ThruStrmPtr;
    m_ThruStrmOwner = false;
    m_ThruStrmPtr = &cs;
  }
  
  /**
   * Set the current time
   *
   * @param t the new current time
   */
  void SetCurTime(double t){
    m_CurTime = t;
  }
  
  /**
   * Get the current time
   *
   * @return the current time
   */
  double GetCurTime() const {
    return m_CurTime;
  }
  
  /**
   * Set the time difference.
   *
   * @param dt the time difference
   */
  void SetDelTime(double dt){
    m_DelTime = dt;
    m_HalfDelTime = 0.5*dt;
  }

  /**
   * Get the time difference.
   *
   * @return the time difference
   */
  double GetDelTime() const {
    return m_DelTime;
  }

  /**
   * Get half of the time difference.
   *
   * @return half of the time difference
   */
  double GetHalfDelTime() const {
    return m_HalfDelTime;
  }


  /**
   * Updates the field for a given time.
   *
   * @param t the new time to update to
   */
  virtual void Update(double t){
    SetDelTime(t - m_CurTime);
    m_CurTime = t;
  }


  virtual void Advance(){
    m_CurTime += m_DelTime;
  }


  virtual void AdvanceSI(const double scale){
    m_CurTime += scale*m_DelTime;
  }


protected:

  /**
   * Check if m_DumpCntr is a multiple of m_DumpPeriod. If not, 
   * return true to indicate we are not dumping now. We are
   * dumping only when m_DumpCntr is a multiple of m_DumpPeriod.
   */
  bool ShouldWeSkipTheDump() { return ++m_DumpCntr%m_DumpPeriod ? true:false; }

  /*** Allow the kids to set the dump period. */
  void SetDumpPeriod(size_t dp) {
    m_DumpPeriod = dp; return;
  }
  


protected:
  /*** The current time of update */
  double m_CurTime;

  /*** The difference between this time and last time */
  double m_DelTime;

  /*** Half of the time step */
  double m_HalfDelTime;

  /*** A stream to send comments to */
  TxThroughStream* m_ThruStrmPtr;

  /*** Whether owner of the thruStrm */
  bool m_ThruStrmOwner;


private:
  /**
   * m_DumpCntr is to be incremented by 1 every time the dump() function of
   * the object that derives from DynObj is called.
   */
  size_t m_DumpCntr;

  /**
   * m_DumpPeriod is initialized by default to 1. It is for use with
   * input file data to set it to a positive value larger than 1 to
   * allow dumping for an object that derives from DynObj only
   * every m_DumpPeriod time when the dump() function of the derived
   * class is called.
   */
  size_t m_DumpPeriod;
};

#endif
