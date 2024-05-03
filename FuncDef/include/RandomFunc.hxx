// --------------------------------------------------------------------
// File:	RandomFunc.h
// Purpose:	Random number generators
//              Interfaces for cross-platform random number generators
// --------------------------------------------------------------------

#ifndef _RandomFunc_HeaderFile
#define _RandomFunc_HeaderFile

#include <TxStreams.h>

// standard includes
#include <stdlib.h>
#include <limits.h>

// vpfunc includes
#include <NoArgFunc.hxx>


// Special way to document functions
/**
 * Get a random number between 0 and 1
 * @return a random number between 0 and 1
 */
Standard_Real randomFunc();

/**
 * Initialize seed for random number generator.
 *
 * @param seed the random number seed
 */
void randomSFunc(size_t seed);


/**
 * This is the Mersenne Twister by Takuji Nishimura and Makoto Matsumoto.
 * It also includes contributions by Shawn Cokus and Matthew Bellew. See
 * accompanying file RandomFunc.cxx for copyright notice and licence.
 */


class RandomFunc : public NoArgFunc
{
 public:
  /*** Default and only constructor */
  RandomFunc() {
    static size_t globalSeed = 0;
    size_t init[4]={0x123, 0x234, 0x345, 0x456}, length=4;
    init[3] += globalSeed;
    init_by_array(init, length);
    globalSeed++;
  };
  
  /*** Destructor */
  virtual ~RandomFunc(){};
  
  /**
   * Set up the functor from data in an attribute set
   *
   * @param tas A TxAttribute set containing the parameters of the functor, which are:
   *
   * tas.getParam("seed") the seed for this random number generator
   */
  virtual void setAttrib(const TxHierAttribSet& tas) throw (TxDebugExcept);
  
  /**
   * Set a seed for this functor (useful for pseudo-random number generators)
   * @param sd the seed
   */
  virtual void setSeed(size_t sd) {
    init_genrand(sd);
  };
  
  inline size_t int32(void) const;
  
  inline Standard_Real real(void) const {
#if (LONG_MAX == 2147483647L)
    return (Standard_Real)(0.5+((long)(int32()^LONG_MIN))*(1.0/4294967296.0));
#else
    return (Standard_Real)((long)int32())*(1.0/4294967296.0);
#endif
  }
  
  // generates a random number on [0,1) with 53-bit resolution
  // This real version is due to Isaku Wada, 2002/01/09 added 
  inline double real53(void) {
    long a = int32() >> 5, b = int32() >> 6;
    return(a*67108864.0+b)*(1.0/9007199254740992.0);
  }
  
  virtual Standard_Real operator()() const {
    return real();
  }
  
 private:
  
  enum { N = 624 };       // length of state vector
  enum { M = 397 };       // period parameter
  mutable size_t state[N], *next;
  mutable int left;
  
  size_t twist(size_t m, size_t u, size_t v) const {
    return m ^ (mixbits(u,v) >> 1) ^ (-(v&1UL) & 0x9908b0dfUL );
  };
  size_t mixbits(size_t u, size_t v) const {
    return ((u) & 0x80000000UL) | ((v) & 0x7fffffffUL);
  };
  
  void init_genrand(size_t s) const;
  void init_by_array(size_t init_key[], size_t key_length);
  void next_state(void) const;
  
 public:
  
  typedef struct {
    size_t state[N];
    int left;
  } state_t;
  
  state_t *getState(void);
  void setState(state_t *st);
};


// generates a random number on [0,0xffffffff]-interval
inline size_t RandomFunc::int32(void) const
{
  size_t y;
  
  if (--left == 0) next_state();
  y = *next++;
  
  // Tempering 
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680UL;
  y ^= (y << 15) & 0xefc60000UL;
  y ^= (y >> 18);
  
  return y;
}


// A global RandomFunc<> object for use with vpRandom<>()
extern RandomFunc theRNG;


inline Standard_Real randomFunc() {
  return (Standard_Real) theRNG.real();
}

inline void randomSFunc(size_t seed) {
  theRNG.setSeed(seed);
}

#endif
