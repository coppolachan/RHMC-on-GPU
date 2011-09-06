    // singleton.h
    #ifndef __SINGLETON_H
    #define __SINGLETON_H

    template <class T>
    class Singleton
    {
    public:
      static T& Instance() {
        static T _instance;
        return _instance;
      }
    private:
      Singleton();          // ctor hidden
      ~Singleton();          // dtor hidden
      Singleton(Singleton const&);    // copy ctor hidden
      Singleton& operator=(Singleton const&);  // assign op hidden
    };

    #endif
    // eof
