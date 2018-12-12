//
//  main.cpp
//  LearnCpp
//
//  Created by Hashu Gill on 2018-10-07.
//  Copyright © 2018 Hashu Gill. All rights reserved.
//
/*
#include <iostream>


struct Node{
    
    Node *next, *prev;
    
    int val;
    
    int key;
    
    Node(Node* p, Node* n, int v, int k):prev(p), next(n), val(v), key(k){};
    
    Node(int v, int k):prev(NULL), next(NULL), val(v), key(k){};
    
    
};


int main() {
    
    Node *nNode;
    
    nNode = new Node(5,6);
    
}

*/

#include <iostream>

using namespace std;

void f( int * ptr ) {
    cout << "The address assigned to the argument ptr is " << ptr << endl;
    
    ptr[0] = 582;
    
    return;
}

int main() {
    int array01[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    
    cout << "The address stored in 'array01' is " << array01[0] << endl;
    
    f( array01 );
    
    int * array02 = new int[10];
    
    cout << "The address stored in 'array02' is " << array02 << endl;
    
    f( array02 );
    
    delete [] array02;
    
    return 0;
}
