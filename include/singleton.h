/*
  singleton.h
  
  Singleton class
  to add singleton properties to class T
*/

#ifndef SINGLETON_H_
#define SINGLETON_H_

template <class T>
class Singleton
{
 public:
  static T& Instance() {
    static T _instance;
    return _instance;
  }
 private:
  Singleton();       // constructor hidden
  ~Singleton();      // destructor hidden
  Singleton(Singleton const&);    // copy constructor hidden
  Singleton& operator=(Singleton const&);  // assign operator hidden
};

#endif //SINGLETON_H_

