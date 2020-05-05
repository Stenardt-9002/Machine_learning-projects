#stream directly from twitter
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

import s1ma as SAVED_VARS

# print(SAVED_VARS.This_test)

class StdOutListener(StreamListener):
    def on_data(self,data):
        print(data)
        pass
    def on_error(self,status):
        print(status)
        pass


string_dick = {
    "pole":"jok1",
    "toch":"spaget"
}



def informationpresent(self, parameter_list):
    pass


class ProceTweets():
    """
    Docstring for class for stream and saving authentication and connection to 
    Twitter Straming API
        def stream_twits(self,fetched_tewwts_fil,hash_list):
    """
    def stream_twits(self,fetched_tewwts_fil,hash_list):
            
        lit_objb = StdOutListener()
        auth_stor = OAuthHandler(SAVED_VARS.CONSUMER_KEY,SAVED_VARS.CONSUMER_SECRET)
        #     ACCESS_TOKEN = ""
        # ACCESS_TOKEN_SECRET = ""
        # CONSUMER_KEY = ""
        # CONSUMER_SECRET = ""
        auth_stor.set_access_token(SAVED_VARS.ACCESS_TOKEN,SAVED_VARS.ACCESS_TOKEN_SECRET)
        stream = Stream(auth_stor,lit_objb)

        # ///////////filter tweets
        stream.filter(track = hash_list)
        pass
    pass





if __name__ == "__main__":
    # for ke1 in string_dick.keys():
    #     print(ke1)
    #     pass

    
    lit_objb = StdOutListener()
    auth_stor = OAuthHandler(SAVED_VARS.CONSUMER_KEY,SAVED_VARS.CONSUMER_SECRET)
#     ACCESS_TOKEN = ""
# ACCESS_TOKEN_SECRET = ""
# CONSUMER_KEY = ""
# CONSUMER_SECRET = ""
    auth_stor.set_access_token(SAVED_VARS.ACCESS_TOKEN,SAVED_VARS.ACCESS_TOKEN_SECRET)
    stream = Stream(auth_stor,lit_objb)

    # ///////////filter tweets
    stream.filter(track = ['akshay','modi'])

